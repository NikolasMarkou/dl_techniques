import keras
from keras import ops
from typing import Dict, List, Optional, Tuple, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger

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
