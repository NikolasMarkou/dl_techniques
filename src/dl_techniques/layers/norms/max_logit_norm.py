"""
MaxLogit Normalization Implementations for Out-of-Distribution Detection.

This module implements variants of MaxLogit normalization for out-of-distribution detection:
1. Basic MaxLogit normalization
2. Decoupled MaxLogit (DML) normalization that separates cosine and L2 components
3. DML+ with dedicated models for cosine and norm components

The implementations follow the paper:
"Decoupling MaxLogit for Out-of-Distribution Detection"

Key Features:
- Improved out-of-distribution detection through logit normalization
- Separation of cosine similarity and L2 norm components
- Support for both combined and specialized model architectures
- Numerically stable implementations with mixed precision support

References:
- Liu, W., et al. "Decoupling MaxLogit for Out-of-Distribution Detection"
"""

import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any, Union, Literal

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MaxLogitNorm(keras.layers.Layer):
    """
    Basic MaxLogit normalization layer for out-of-distribution detection.

    This layer applies L2 normalization to logits, which has been shown to improve
    out-of-distribution detection by normalizing the magnitude component while
    preserving the directional information in the logits.

    The layer computes: output = inputs / ||inputs||_2

    This normalization helps separate in-distribution and out-of-distribution
    samples by focusing on the angular relationships rather than magnitude
    differences in the logit space.

    Args:
        axis: Integer, the axis along which to normalize. Typically -1 for the
            feature dimension. Must be a valid axis for the input tensor.
            Defaults to -1.
        epsilon: Float, small constant added for numerical stability during
            division. Must be positive. Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary tensor. Most commonly used with 2D logit tensors of shape
        ``(batch_size, num_classes)`` from classification layers.

    Output shape:
        Same as input shape. The tensor is L2 normalized along the specified axis.

    Raises:
        ValueError: If epsilon is not positive.

    Example:
        Basic usage for OOD detection in classification:

        ```python
        import keras
        from dl_techniques.layers.norms.max_logit_norm import MaxLogitNorm

        # Apply to classification logits
        inputs = keras.Input(shape=(10,))  # 10 classes
        logits = keras.layers.Dense(10)(inputs)
        normalized_logits = MaxLogitNorm(axis=-1, epsilon=1e-7)(logits)

        # Use normalized logits for both classification and OOD detection
        predictions = keras.layers.Softmax()(normalized_logits)
        model = keras.Model(inputs, [predictions, normalized_logits])
        ```

        Integration in a complete OOD detection system:

        ```python
        def create_ood_detection_model(num_classes=10, hidden_dim=512):
            inputs = keras.Input(shape=(784,))  # MNIST-like input

            # Feature extraction
            x = keras.layers.Dense(hidden_dim, activation='relu')(inputs)
            x = keras.layers.Dropout(0.5)(x)
            x = keras.layers.Dense(hidden_dim, activation='relu')(x)
            x = keras.layers.Dropout(0.5)(x)

            # Classification logits
            logits = keras.layers.Dense(num_classes, name='logits')(x)

            # Apply MaxLogit normalization for OOD detection
            normalized_logits = MaxLogitNorm(
                axis=-1,
                epsilon=1e-7,
                name='maxlogit_norm'
            )(logits)

            # Standard classification output
            predictions = keras.layers.Softmax(name='predictions')(normalized_logits)

            return keras.Model(
                inputs=inputs,
                outputs={
                    'predictions': predictions,
                    'normalized_logits': normalized_logits,
                    'raw_logits': logits
                }
            )

        model = create_ood_detection_model()

        # During inference, use max(normalized_logits) as OOD score
        # Higher scores indicate in-distribution samples
        ```

        Custom epsilon for different precision requirements:

        ```python
        # For mixed precision training
        maxlogit_fp16 = MaxLogitNorm(axis=-1, epsilon=1e-5)

        # For high precision requirements
        maxlogit_precise = MaxLogitNorm(axis=-1, epsilon=1e-10)
        ```

    Note:
        This implementation follows the modern Keras 3 pattern and is optimized for
        numerical stability across different precision modes. The layer maintains
        the same computational graph structure during training and inference.
    """

    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs early in __init__
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration parameters
        self.axis = axis
        self.epsilon = epsilon

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Apply MaxLogit normalization to inputs.

        Args:
            inputs: Input logits tensor to be normalized.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                kept for consistency.

        Returns:
            L2 normalized tensor with the same shape as inputs.
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
            Output shape tuple (same as input shape for normalization layers).
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all constructor arguments needed to recreate
            this layer instance.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DecoupledMaxLogit(keras.layers.Layer):
    """
    Decoupled MaxLogit (DML) normalization layer for enhanced OOD detection.

    This layer separates MaxLogit into cosine similarity and L2 norm components
    with learnable weighting between them. This decomposition allows better
    understanding and control of which component contributes most to
    out-of-distribution detection.

    The layer computes:
    1. normalized = inputs / ||inputs||_2  (cosine component)
    2. max_cosine = max(normalized)
    3. max_norm = max(||inputs||_2)
    4. combined_score = constant * max_cosine + max_norm

    This separation enables analysis of whether OOD detection is driven by
    angular differences (cosine) or magnitude differences (norm).

    Args:
        constant: Float, weight coefficient for combining cosine and L2 components.
            Higher values emphasize the cosine component. Must be positive.
            Defaults to 1.0.
        axis: Integer, the axis along which to compute normalization and maximum values.
            Typically -1 for the feature dimension. Defaults to -1.
        epsilon: Float, small constant added for numerical stability during
            division. Must be positive. Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary tensor. Most commonly used with 2D logit tensors of shape
        ``(batch_size, num_classes)`` from classification layers.

    Output shape:
        Returns a tuple of three tensors, each with shape obtained by removing
        the normalized axis from the input shape:
        - Combined score: (batch_size,) for input (batch_size, features)
        - MaxCosine component: (batch_size,)
        - MaxNorm component: (batch_size,)

    Raises:
        ValueError: If constant or epsilon is not positive.

    Example:
        Basic usage for analyzing OOD detection components:

        ```python
        import keras
        from dl_techniques.layers.norms.max_logit_norm import DecoupledMaxLogit

        # Create layer with custom weighting
        dml_layer = DecoupledMaxLogit(
            constant=2.0,  # Emphasize cosine component
            axis=-1,
            epsilon=1e-7
        )

        # Apply to logits
        inputs = keras.Input(shape=(10,))  # 10 classes
        logits = keras.layers.Dense(10)(inputs)

        # Get all three components
        combined_score, max_cosine, max_norm = dml_layer(logits)

        model = keras.Model(
            inputs=inputs,
            outputs={
                'combined_score': combined_score,
                'max_cosine': max_cosine,
                'max_norm': max_norm
            }
        )
        ```

        Complete OOD detection system with component analysis:

        ```python
        def create_dml_ood_model(num_classes=10, constant=1.5):
            inputs = keras.Input(shape=(784,))

            # Feature extraction backbone
            x = keras.layers.Dense(512, activation='relu')(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)

            x = keras.layers.Dense(256, activation='relu')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)

            # Classification head
            logits = keras.layers.Dense(num_classes, name='logits')(x)

            # Standard predictions for classification
            predictions = keras.layers.Softmax(name='predictions')(logits)

            # DML analysis for OOD detection
            combined_score, max_cosine, max_norm = DecoupledMaxLogit(
                constant=constant,
                axis=-1,
                name='dml_analysis'
            )(logits)

            return keras.Model(
                inputs=inputs,
                outputs={
                    'predictions': predictions,
                    'ood_score': combined_score,
                    'cosine_component': max_cosine,
                    'norm_component': max_norm,
                    'raw_logits': logits
                }
            )

        # Train model normally for classification
        model = create_dml_ood_model(num_classes=10, constant=1.2)
        model.compile(
            optimizer='adam',
            loss={'predictions': 'sparse_categorical_crossentropy'},
            metrics={'predictions': 'accuracy'}
        )

        # During inference, analyze which component drives OOD detection
        results = model.predict(test_data)
        cosine_scores = results['cosine_component']
        norm_scores = results['norm_component']
        combined_scores = results['ood_score']
        ```

        Hyperparameter analysis for optimal weighting:

        ```python
        # Experiment with different constant values
        constants_to_try = [0.5, 1.0, 1.5, 2.0, 3.0]

        for constant in constants_to_try:
            dml = DecoupledMaxLogit(constant=constant)
            # Evaluate OOD detection performance
            # Choose constant that maximizes AUROC
        ```

    Note:
        The constant parameter can be tuned based on the specific dataset and
        OOD detection requirements. Higher values emphasize angular differences,
        while lower values emphasize magnitude differences.
    """

    def __init__(
        self,
        constant: float = 1.0,
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs early in __init__
        if constant <= 0:
            raise ValueError(f"constant must be positive, got {constant}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration parameters
        self.constant = constant
        self.axis = axis
        self.epsilon = epsilon

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Tuple[keras.KerasTensor, keras.KerasTensor, keras.KerasTensor]:
        """
        Apply decoupled MaxLogit normalization.

        Args:
            inputs: Input logits tensor to be analyzed.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                kept for consistency.

        Returns:
            Tuple of three tensors:
            - Combined score: constant * max_cosine + max_norm
            - MaxCosine component: max(normalized_inputs)
            - MaxNorm component: max(||inputs||_2)
        """
        # Cast inputs to computation dtype for numerical stability
        inputs = ops.cast(inputs, self.compute_dtype)

        # Compute L2 norm with numerical stability
        squared = ops.square(inputs)
        norm = ops.sqrt(
            ops.sum(squared, axis=self.axis, keepdims=True) + self.epsilon
        )

        # Compute normalized features (cosine component)
        normalized = inputs / norm

        # Get maximum cosine similarity
        max_cosine = ops.max(normalized, axis=self.axis)

        # Get maximum norm (squeeze to remove keepdims)
        max_norm = ops.squeeze(ops.max(norm, axis=self.axis), axis=-1)

        # Combine with learned weight
        combined_score = self.constant * max_cosine + max_norm

        return combined_score, max_cosine, max_norm

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Tuple[Optional[int], ...], ...]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Tuple of three output shape tuples for (combined, max_cosine, max_norm).
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Remove the axis dimension to get output shape
        if self.axis == -1 or self.axis == len(input_shape_list) - 1:
            output_shape = tuple(input_shape_list[:-1])
        else:
            # Handle positive axis values
            axis_pos = self.axis if self.axis >= 0 else len(input_shape_list) + self.axis
            output_shape_list = input_shape_list[:axis_pos] + input_shape_list[axis_pos + 1:]
            output_shape = tuple(output_shape_list)

        # All three outputs have the same shape
        return (output_shape, output_shape, output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all constructor arguments needed to recreate
            this layer instance.
        """
        config = super().get_config()
        config.update({
            "constant": self.constant,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DMLPlus(keras.layers.Layer):
    """
    DML+ implementation for separate focal and center models.

    This layer is designed to be used in architectures where separate models
    are optimized for different components of the decoupled MaxLogit. The focal
    model focuses on the cosine similarity component (MaxCosine) while the center
    model focuses on the magnitude component (MaxNorm).

    This separation allows for specialized optimization of each component,
    potentially leading to better overall OOD detection performance through
    ensemble methods or specialized training procedures.

    Model Types:
    - **Focal Model**: Optimized for angular relationships, returns MaxCosine scores
    - **Center Model**: Optimized for magnitude relationships, returns MaxNorm scores and normalization factors

    Args:
        model_type: Literal string, either "focal" or "center" indicating the
            specialized model type. This determines the output format and
            computation focus.
        axis: Integer, the axis along which to compute normalization and maximum values.
            Typically -1 for the feature dimension. Defaults to -1.
        epsilon: Float, small constant added for numerical stability during
            division. Must be positive. Defaults to 1e-7.
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        Arbitrary tensor. Most commonly used with 2D logit tensors of shape
        ``(batch_size, num_classes)`` from classification layers.

    Output shape:
        Depends on model_type:
        - **Focal model**: Shape with axis dimension removed, e.g., (batch_size,) for input (batch_size, features)
        - **Center model**: Tuple of two tensors:
          - MaxNorm scores: (batch_size,)
          - Normalization factors: (batch_size, 1) or original keepdims shape

    Raises:
        ValueError: If model_type is not "focal" or "center", or if epsilon is not positive.

    Example:
        Creating separate focal and center models:

        ```python
        import keras
        from dl_techniques.layers.norms.max_logit_norm import DMLPlus

        def create_focal_model(num_classes=10):
            \"\"\"Model optimized for cosine similarity component.\"\"\"
            inputs = keras.Input(shape=(784,))

            # Shared feature extraction
            x = keras.layers.Dense(512, activation='relu')(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)

            # Focal-optimized layers (emphasize angular separability)
            x = keras.layers.Dense(256, activation='relu')(x)
            x = keras.layers.LayerNormalization()(x)  # Normalize features

            # Classification logits
            logits = keras.layers.Dense(num_classes)(x)

            # Focal output (MaxCosine)
            max_cosine = DMLPlus(
                model_type="focal",
                axis=-1,
                name='focal_output'
            )(logits)

            # Also provide standard predictions
            predictions = keras.layers.Softmax()(logits)

            return keras.Model(
                inputs=inputs,
                outputs={
                    'predictions': predictions,
                    'focal_score': max_cosine,
                    'logits': logits
                }
            )

        def create_center_model(num_classes=10):
            inputs = keras.Input(shape=(784,))

            # Shared feature extraction
            x = keras.layers.Dense(512, activation='relu')(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(0.3)(x)

            # Center-optimized layers (emphasize magnitude separability)
            x = keras.layers.Dense(256, activation='relu')(x)
            # No normalization to preserve magnitude information

            # Classification logits
            logits = keras.layers.Dense(num_classes)(x)

            # Center output (MaxNorm + normalization factor)
            max_norm, norm_factor = DMLPlus(
                model_type="center",
                axis=-1,
                name='center_output'
            )(logits)

            # Standard predictions
            predictions = keras.layers.Softmax()(logits)

            return keras.Model(
                inputs=inputs,
                outputs={
                    'predictions': predictions,
                    'center_score': max_norm,
                    'norm_factor': norm_factor,
                    'logits': logits
                }
            )

        # Create specialized models
        focal_model = create_focal_model()
        center_model = create_center_model()
        ```

        Ensemble approach using both models:

        ```python
        def create_dml_plus_ensemble():
            \"\"\"Combine focal and center models for enhanced OOD detection.\"\"\"
            inputs = keras.Input(shape=(784,))

            # Get outputs from both specialized models
            focal_outputs = focal_model(inputs)
            center_outputs = center_model(inputs)

            # Combine scores with learnable weights
            combined_score = keras.layers.Dense(1, activation='linear', name='ensemble')(
                keras.layers.Concatenate()([
                    keras.layers.Expand_dims()(focal_outputs['focal_score']),
                    keras.layers.Expand_dims()(center_outputs['center_score'])
                ])
            )

            # Use average of predictions for classification
            avg_predictions = keras.layers.Average()([
                focal_outputs['predictions'],
                center_outputs['predictions']
            ])

            return keras.Model(
                inputs=inputs,
                outputs={
                    'predictions': avg_predictions,
                    'ood_score': combined_score,
                    'focal_component': focal_outputs['focal_score'],
                    'center_component': center_outputs['center_score']
                }
            )

        ensemble_model = create_dml_plus_ensemble()
        ```

        Training strategy for specialized models:

        ```python
        # Different loss functions for different models
        focal_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'predictions': 'sparse_categorical_crossentropy',
                # Could add auxiliary losses for focal optimization
            },
            metrics={'predictions': 'accuracy'}
        )

        center_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'predictions': 'sparse_categorical_crossentropy',
                # Could add auxiliary losses for center optimization
            },
            metrics={'predictions': 'accuracy'}
        )
        ```

    Note:
        The DMLPlus approach allows for more specialized optimization of each
        component. Training procedures can be tailored to enhance either the
        angular (focal) or magnitude (center) discrimination capabilities.
    """

    def __init__(
        self,
        model_type: Literal["focal", "center"],
        axis: int = -1,
        epsilon: float = 1e-7,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs early in __init__
        if model_type not in ["focal", "center"]:
            raise ValueError(f"model_type must be 'focal' or 'center', got {model_type}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        # Store configuration parameters
        self.model_type = model_type
        self.axis = axis
        self.epsilon = epsilon

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Apply DML+ normalization based on model type.

        Args:
            inputs: Input logits tensor to be processed.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                kept for consistency.

        Returns:
            For focal model: MaxCosine score tensor.
            For center model: Tuple of (MaxNorm score, normalization factor).
        """
        # Cast inputs to computation dtype for numerical stability
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
            # Center model returns MaxNorm and normalization factor
            max_norm = ops.squeeze(ops.max(norm, axis=self.axis), axis=-1)
            return max_norm, norm

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Union[Tuple[Optional[int], ...], Tuple[Tuple[Optional[int], ...], ...]]:
        """
        Compute the output shape of the layer.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape(s) depending on model type:
            - Focal model: Single shape tuple with axis dimension removed
            - Center model: Tuple of two shape tuples (max_norm, norm_factor)
        """
        # Convert to list for manipulation
        input_shape_list = list(input_shape)

        # Remove the axis dimension for reduced shape
        if self.axis == -1 or self.axis == len(input_shape_list) - 1:
            reduced_shape = tuple(input_shape_list[:-1])
        else:
            # Handle positive axis values
            axis_pos = self.axis if self.axis >= 0 else len(input_shape_list) + self.axis
            reduced_shape_list = input_shape_list[:axis_pos] + input_shape_list[axis_pos + 1:]
            reduced_shape = tuple(reduced_shape_list)

        if self.model_type == "focal":
            return reduced_shape
        else:
            # Center model returns (max_norm, norm_factor)
            # norm_factor keeps the original shape with keepdims=True
            return (reduced_shape, input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            Dictionary containing all constructor arguments needed to recreate
            this layer instance.
        """
        config = super().get_config()
        config.update({
            "model_type": self.model_type,
            "axis": self.axis,
            "epsilon": self.epsilon
        })
        return config

# ---------------------------------------------------------------------