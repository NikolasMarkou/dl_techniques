"""
Selectively route tensors for conditional training or data imputation.

This layer acts as a data-driven multiplexer, choosing between two input
tensors (`ground_truth` and `inference`) on a sample-by-sample basis within
a batch. Its primary function is to enable sophisticated training schemes,
particularly in semi-supervised or generative modeling contexts, where
different samples in a batch require different computational paths or loss
treatments.

Architecture:
    The layer's design is a conditional switch. It takes two tensors of
    identical shape as input. For each sample in the batch, it inspects the
    `ground_truth` tensor. If every element of that sample is zero, it routes
    the corresponding sample from the `inference` tensor to the output.
    Otherwise, if the `ground_truth` sample contains at least one non-zero
    element, it routes the `ground_truth` sample itself to the output.

    This mechanism is a powerful tool for masking the loss function for
    certain samples. When this layer's output is fed into a loss function,
    the behavior is bifurcated:
    1.  **"Unlabeled" samples (all-zero ground truth):** The output is the
        model's inference. The loss is computed on the model's prediction,
        allowing gradients to flow back and train the upstream network.
    2.  **"Labeled" samples (non-zero ground truth):** The output is the
        ground truth itself. If the loss function's target is also the ground
        truth, the resulting loss will be zero (`L(gt, gt) = 0`), effectively
        preventing any gradients from these samples from affecting the
        upstream inference network.

Foundational Mathematics and Algorithm:
    The core of the layer is a conditional selection operation, typically
    implemented with a `where` clause:
    `output = where(condition, inference, ground_truth)`

    The `condition` is a boolean tensor derived from the `ground_truth` input.
    The derivation process for each sample in the batch is as follows:
    1.  An element-wise comparison `is_zero = (ground_truth == 0.0)` is performed.
    2.  A reduction using the logical `all` operator is applied across all
        non-batch dimensions (e.g., height, width, channels) of `is_zero`.
        This yields a boolean vector of shape `(batch_size,)`, where each
        element is `True` if the corresponding sample was all zeros.
    3.  This boolean vector is then broadcast back to the rank of the input
        tensors by adding singleton dimensions. This allows the `where`
        operation to perform the element-wise selection correctly across
        the entire sample.

References:
    This layer implements a common design pattern rather than a specific, citable
    algorithm. The underlying principle is fundamental to various advanced machine
    learning techniques, including:

    1.  **Semi-Supervised Learning:** Where a model is trained on a dataset
        containing both labeled and unlabeled examples. This layer provides a
        mechanism to apply a supervised loss to labeled data and an
        unsupervised or consistency loss to unlabeled data within a single
        training step.
    2.  **Generative Modeling and Inpainting:** In tasks like image completion,
        this pattern can be used to ensure the loss is only calculated on the
        missing or "generated" regions of an image, while known regions (the
        "ground truth") contribute zero loss.
    3.  **Masked Modeling:** Conceptually related to techniques in self-supervised
        learning (e.g., Masked Autoencoders), where parts of the input are
        selectively processed or ignored by the loss function.
"""

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
