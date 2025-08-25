import keras
from typing import List, Tuple, Optional, Any

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class ConditionalOutputLayer(keras.layers.Layer):
    """
    A custom layer for conditional output selection based on ground truth values.

    This layer implements a batch-wise selection mechanism where for each item in the batch:
    - If the ground truth tensor item contains all zeros, output the inference tensor item
    - Otherwise, output the ground truth tensor item

    The layer maintains the same shape and dtype as the input tensors and provides
    a clean abstraction for conditional training scenarios.

    Key Features:
        - Batch-wise conditional selection based on zero detection
        - Shape and dtype preservation
        - Efficient broadcasting for multi-dimensional tensors
        - Full serialization support

    Args:
        **kwargs: Additional keyword arguments for the Layer base class.

    Input shape:
        List of two tensors with identical shapes:
        - ground_truth: Tensor with shape (batch_size, ...)
        - inference: Tensor with shape (batch_size, ...)

    Output shape:
        Tensor with same shape as input tensors: (batch_size, ...)

    Example:
        ```python
        # Create layer
        layer = ConditionalOutputLayer()

        # Example tensors
        ground_truth = keras.ops.array([[0., 0.], [1., 2.]])  # First batch item all zeros
        inference = keras.ops.array([[3., 4.], [5., 6.]])

        # Apply conditional selection
        output = layer([ground_truth, inference])
        # Result: [[3., 4.], [1., 2.]]  # First item from inference, second from ground_truth

        # In a model
        gt_input = keras.Input(shape=(10,), name='ground_truth')
        inf_input = keras.Input(shape=(10,), name='inference')
        output = ConditionalOutputLayer()([gt_input, inf_input])
        model = keras.Model(inputs=[gt_input, inf_input], outputs=output)
        ```

    Note:
        Both input tensors must have identical shapes and compatible dtypes.
        The layer efficiently handles multi-dimensional tensors by reducing
        across all non-batch dimensions for zero detection.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the BatchConditionalOutputLayer.

        Args:
            **kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(
        self,
        inputs: List[keras.KerasTensor],
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass of the layer.

        Performs batch-wise conditional selection where each batch item is selected
        from either ground_truth or inference based on whether ground_truth contains
        all zeros.

        Args:
            inputs: List containing [ground_truth, inference] tensors with identical shapes.
            training: Boolean indicating training mode (unused but included for consistency).

        Returns:
            keras.KerasTensor: Output tensor with values selected from either ground truth
                             or inference based on ground truth zero detection.

        Raises:
            ValueError: If inputs list doesn't contain exactly 2 tensors or if shapes don't match.
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError(
                f"BatchConditionalOutputLayer expects a list of exactly 2 tensors, "
                f"got {type(inputs)} with length {len(inputs) if hasattr(inputs, '__len__') else 'unknown'}"
            )

        ground_truth, inference = inputs

        # Validate shapes match
        if ground_truth.shape != inference.shape:
            raise ValueError(
                f"Input tensor shapes must match exactly. "
                f"Got ground_truth: {ground_truth.shape}, inference: {inference.shape}"
            )

        # Check for all zeros per batch item
        # Reduce across all dimensions except batch (axis 0)
        reduction_axes = list(range(1, len(ground_truth.shape)))

        # Detect if each batch item in ground_truth is all zeros
        is_all_zeros = keras.ops.all(
            keras.ops.equal(ground_truth, 0.0),
            axis=reduction_axes if reduction_axes else None
        )

        # Reshape condition for proper broadcasting to original tensor shape
        # Add singleton dimensions for all non-batch axes
        broadcast_shape = [-1] + [1] * (len(ground_truth.shape) - 1)
        is_all_zeros_broadcasted = keras.ops.reshape(is_all_zeros, broadcast_shape)

        # Conditional selection: if ground_truth is all zeros, use inference; otherwise use ground_truth
        output = keras.ops.where(is_all_zeros_broadcasted, inference, ground_truth)

        return output

    def compute_output_shape(
        self,
        input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the layer's output shape.

        Args:
            input_shape: List of input shapes for [ground_truth, inference].

        Returns:
            Tuple representing the output shape (same as input shapes).

        Raises:
            ValueError: If input_shape doesn't contain exactly 2 shapes or shapes don't match.
        """
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                f"Expected list of 2 input shapes, got {type(input_shape)} "
                f"with length {len(input_shape) if hasattr(input_shape, '__len__') else 'unknown'}"
            )

        ground_truth_shape, inference_shape = input_shape

        if ground_truth_shape != inference_shape:
            raise ValueError(
                f"Input shapes must be identical. "
                f"Got ground_truth: {ground_truth_shape}, inference: {inference_shape}"
            )

        return ground_truth_shape

    def get_config(self) -> dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns:
            dict: Layer configuration dictionary containing all necessary
                 parameters for reconstruction.
        """
        config = super().get_config()

        return config

# ---------------------------------------------------------------------
