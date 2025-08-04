"""
This module provides a `OneHotEncoding` layer, a custom Keras layer that performs
one-hot encoding for multiple categorical features simultaneously.

One-hot encoding is a standard and essential preprocessing step for machine learning models
that work with categorical data. It converts integer-based categorical labels
(e.g., 0, 1, 2) into a binary vector format where only one bit is "hot" (set to 1),
representing the specific category. This prevents the model from assuming a false
ordinal relationship between categories (i.e., that category 2 is "greater" than
category 1).

While this operation can be done in an offline preprocessing step, implementing it
as a Keras layer offers significant advantages, which this implementation provides.

Key Features and Architectural Benefits:

1.  **Model Encapsulation:**
    -   By making one-hot encoding a layer, the preprocessing logic becomes an integral
        part of the model itself.
    -   This simplifies deployment and inference pipelines, as the model is self-contained.
        You can feed it raw categorical integers, and it will handle the conversion
        internally, eliminating the need to manage a separate preprocessing step in
        production.

2.  **Multi-Feature Handling:**
    -   This layer is explicitly designed to handle tabular data with multiple
        categorical columns. The `cardinalities` list directly corresponds to the
        number of unique categories in each input feature column.
    -   It processes all features in a single `call`, making it a clean and unified
        interface for tabular data.

3.  **Efficient and Vectorized Operation:**
    -   The layer iterates through each categorical feature and applies the backend-agnostic
        and highly optimized `keras.ops.one_hot` function.
    -   The resulting one-hot vectors for each feature are then efficiently combined
        using a single `keras.ops.concatenate` operation at the end. This approach
        leverages the performance of the underlying backend (TensorFlow, JAX, or PyTorch)
        for fast, vectorized computation.

The operational flow is as follows:
-   The layer is initialized with a list of `cardinalities`, one for each input feature.
-   During the forward pass, it takes a batch of integer-encoded data of shape `(batch_size, num_features)`.
-   It loops through each feature column, creating a one-hot vector of the specified cardinality.
-   Finally, it concatenates all these vectors along the last axis to produce a single, wide one-hot encoded tensor.
"""

import keras
from keras import ops
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------

class OneHotEncoding(keras.layers.Layer):
    """One-hot encoding layer for categorical features with enhanced efficiency.

    This layer efficiently converts categorical features to one-hot encoded representations
    using vectorized operations and proper memory management.

    Args:
        cardinalities: List of cardinalities for each categorical feature.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            cardinalities: List[int],
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.cardinalities = cardinalities
        self.total_dim = sum(cardinalities)
        self.cumulative_cardinalities = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by computing cumulative cardinalities for efficient indexing."""
        if self.cardinalities:
            # Precompute cumulative cardinalities for efficient slicing
            self.cumulative_cardinalities = [0]
            for card in self.cardinalities:
                self.cumulative_cardinalities.append(self.cumulative_cardinalities[-1] + card)
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply one-hot encoding to categorical inputs.

        Args:
            inputs: Categorical input tensor of shape (batch_size, n_cat_features).

        Returns:
            One-hot encoded tensor of shape (batch_size, total_categorical_dim).
        """
        if len(self.cardinalities) == 0:
            batch_size = ops.shape(inputs)[0]
            return ops.zeros((batch_size, 0), dtype=self.compute_dtype)

        # Convert to int32 for one_hot operation
        inputs_int = ops.cast(inputs, "int32")

        outputs = []
        for i, cardinality in enumerate(self.cardinalities):
            # Extract the i-th categorical feature efficiently
            cat_feature = inputs_int[:, i]

            # One-hot encode with proper dtype
            one_hot = ops.one_hot(
                cat_feature,
                cardinality,
                dtype=self.compute_dtype
            )
            outputs.append(one_hot)

        # Concatenate all one-hot encodings efficiently
        if outputs:
            return ops.concatenate(outputs, axis=-1)
        else:
            batch_size = ops.shape(inputs)[0]
            return ops.zeros((batch_size, 0), dtype=self.compute_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int]:
        """Compute the output shape of the layer."""
        return (input_shape[0], self.total_dim)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            "cardinalities": self.cardinalities,
        })
        return config

# ---------------------------------------------------------------------
