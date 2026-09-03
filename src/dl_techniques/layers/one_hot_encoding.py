"""
OneHotEncoding, a layer that one-hot encodes several categorical columns.

Converts a batch of integer-encoded categorical features into a single wide
binary tensor, avoiding the false ordinal relationship a raw integer label
implies. Each feature column is one-hot encoded independently with its own
cardinality, then the results are concatenated along the last axis. Putting
this inside the model graph means a caller feeds raw category integers and
never needs a separate preprocessing step at inference time.

The layer is initialized with a list of cardinalities, one per input feature
column, and expects an input of shape `(batch_size, num_features)`.
"""

import keras
from keras import ops
from typing import Dict, List, Optional, Tuple, Any

from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.layers.one_hot_encoding")
class OneHotEncoding(keras.layers.Layer):
    """
    One-hot encoding layer for multiple categorical features.

    This layer converts integer-encoded categorical features into binary
    one-hot vectors and concatenates them into a single wide tensor. It
    handles multiple features simultaneously via a list of per-feature
    cardinalities, applying the backend-optimised ``keras.ops.one_hot``
    independently to each column and joining the results along the last
    axis. Encapsulating the encoding inside the model graph eliminates
    the need for a separate preprocessing step at inference time.

    The forward computation for a batch of ``F`` features with
    cardinalities ``(c_1, c_2, ..., c_F)`` is:
    ``y = concat(one_hot(x[:, 0], c_1), ..., one_hot(x[:, F-1], c_F))``,
    producing an output of width ``sum(c_i)``.

    Architecture:

    .. code-block:: text

        ┌──────────────────────────────────┐
        │  Input [batch, num_features]     │
        │  (integer-encoded categoricals)  │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Cast to int32                   │
        └──────────────┬───────────────────┘
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
        ┌────────┐ ┌────────┐ ┌────────┐
        │one_hot │ │one_hot │ │one_hot │
        │(c_1)   │ │(c_2)   │ │(c_F)   │
        └───┬────┘ └───┬────┘ └───┬────┘
            │          │          │
            └──────────┼──────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Concatenate along last axis     │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Output [batch, sum(c_i)]        │
        └──────────────────────────────────┘

    :param cardinalities: List of integers giving the number of unique
        categories for each input feature column.
    :type cardinalities: List[int]
    :param kwargs: Additional keyword arguments for the Layer base class.
    :type kwargs: Any"""

    def __init__(
            self,
            cardinalities: List[int],
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        # An empty list is allowed and gives a zero-width output.
        if any(card <= 0 for card in cardinalities):
            raise ValueError(
                f"All cardinalities must be positive integers, got {cardinalities}"
            )

        self.cardinalities = cardinalities
        self.total_dim = sum(cardinalities)
        self.cumulative_cardinalities = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the layer by pre-computing cumulative cardinalities.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]"""
        if self.cardinalities:
            self.cumulative_cardinalities = [0]
            for card in self.cardinalities:
                self.cumulative_cardinalities.append(self.cumulative_cardinalities[-1] + card)
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply one-hot encoding to categorical inputs.

        :param inputs: Categorical input tensor of shape
            ``(batch_size, n_cat_features)``.
        :type inputs: Any
        :return: One-hot encoded tensor of shape
            ``(batch_size, total_categorical_dim)``.
        :rtype: Any"""
        if len(self.cardinalities) == 0:
            batch_size = ops.shape(inputs)[0]
            return ops.zeros((batch_size, 0), dtype=self.compute_dtype)

        inputs_int = ops.cast(inputs, "int32")

        outputs = []
        for i, cardinality in enumerate(self.cardinalities):
            cat_feature = inputs_int[:, i]
            one_hot = ops.one_hot(
                cat_feature,
                cardinality,
                dtype=self.compute_dtype
            )
            outputs.append(one_hot)

        if outputs:
            return ops.concatenate(outputs, axis=-1)
        else:
            batch_size = ops.shape(inputs)[0]
            return ops.zeros((batch_size, 0), dtype=self.compute_dtype)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], int]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple.
        :rtype: Tuple[Optional[int], int]"""
        return (input_shape[0], self.total_dim)

    def get_config(self) -> Dict[str, Any]:
        """Return layer configuration for serialization.

        :return: Dictionary containing all constructor parameters.
        :rtype: Dict[str, Any]"""
        config = super().get_config()
        config.update({
            "cardinalities": self.cardinalities,
        })
        return config
