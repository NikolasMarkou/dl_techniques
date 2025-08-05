"""
Batch Conditional Output Layer
============================

A custom layer that implements batch-wise conditional selection between ground truth
and inference outputs. This layer is particularly useful in scenarios where you want
to conditionally mix ground truth and model predictions during training or inference.

Key Features:
------------
- Batch-wise conditional selection
- Zero-detection based switching
- Shape-preserving operations
- Broadcasting support
- Tensor shape validation

Use Cases:
---------
- Teacher forcing in sequence models
- Curriculum learning
- Mixed ground truth/inference training
- Conditional inference pipelines
- Progressive model deployment

Architecture:
------------
The layer implements a conditional selection mechanism:
1. Detects all-zero elements in ground truth per batch item
2. Creates a boolean mask for selection
3. Selectively outputs either ground truth or inference values

The computation flow is:
[ground_truth, inference] -> zero_detection -> mask_creation -> conditional_select -> output

Selection Logic:
--------------
For each item in the batch:
- If ground truth contains all zeros: Use inference output
- Otherwise: Use ground truth value

This allows for flexible batch-wise mixing of sources.

Usage Examples:
-------------
```python
# Basic usage
ground_truth = layers.Input(shape=(height, width, channels))
inference = model(inputs)
output = BatchConditionalOutputLayer()([ground_truth, inference])

# In a model
model = keras.Model(
    inputs=[model_input, ground_truth],
    outputs=output
)

# With dynamic tensors
output = BatchConditionalOutputLayer()(
    [ground_truth_tensor, inference_tensor]
)
```

Implementation Notes:
------------------
- Expects exactly two inputs of same shape
- Performs shape validation
- Preserves input shapes in output
- Handles arbitrary tensor dimensions
- Optimized for broadcast operations
"""

import keras
import tensorflow as tf
from keras import layers
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------

@keras.utils.register_keras_serializable()
class BatchConditionalOutputLayer(layers.Layer):
    """A layer that selectively outputs ground truth or inference values per batch item.

    For each item in the batch, the layer selects between ground truth and inference
    based on whether the ground truth item contains all zeros:
    - If ground truth is all zeros: Use inference value
    - Otherwise: Use ground truth value

    The layer maintains tensor shapes and supports arbitrary dimensions.

    Args:
        name: Optional name for the layer
        **kwargs: Additional layer keyword arguments

    Example:
        ```python
        layer = BatchConditionalOutputLayer()
        output = layer([ground_truth, inference])
        ```
    """

    def __init__(
            self,
            name: Optional[str] = None,
            **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)

    def call(
            self,
            inputs: List[tf.Tensor]
    ) -> tf.Tensor:
        """Forward pass implementing the conditional selection logic.

        Args:
            inputs: List containing [ground_truth, inference] tensors
                Both tensors must have identical shapes

        Returns:
            Tensor where each batch item is selected from either
            ground truth or inference based on the selection logic

        Raises:
            ValueError: If inputs list doesn't contain exactly 2 tensors
            ValueError: If input tensor shapes don't match
        """
        # Validate inputs
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError(
                f"Expected 2 input tensors, got {len(inputs)}"
            )

        ground_truth, inference = inputs

        # Verify shape compatibility
        if ground_truth.shape != inference.shape:
            raise ValueError(
                f"Input shapes must match. Got {ground_truth.shape} "
                f"and {inference.shape}"
            )

        # Create selection mask for all-zero items
        # Reduce across all dimensions except batch (axis 0)
        selection_mask = tf.reduce_all(
            tf.equal(ground_truth, 0),
            axis=list(range(1, len(ground_truth.shape)))
        )

        # Reshape mask for broadcasting
        broadcast_shape = [-1] + [1] * (len(ground_truth.shape) - 1)
        selection_mask = tf.reshape(selection_mask, broadcast_shape)

        # Select outputs using mask
        return tf.where(selection_mask, inference, ground_truth)

    def compute_output_shape(
            self,
            input_shape: List[Tuple[Optional[int], ...]]
    ) -> Tuple[Optional[int], ...]:
        """Computes the output shape based on input shapes.

        Args:
            input_shape: List of shapes for [ground_truth, inference]

        Returns:
            Output tensor shape (same as input shapes)

        Raises:
            ValueError: If input_shape is invalid
        """
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError(
                f"Expected 2 input shapes, got {len(input_shape)}"
            )
        return input_shape[0]

# ---------------------------------------------------------------------

