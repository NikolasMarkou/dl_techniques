"""
Vector squashing non-linearity for Capsule Networks.

This layer applies a specific non-linear function designed to operate on
vectors (capsules) rather than scalars. Its primary architectural purpose
in a Capsule Network is to normalize the length of a capsule's output
vector to lie within the range [0, 1), making the length interpretable as
the probability that the entity represented by the capsule exists. Crucially,
this is achieved while preserving the vector's orientation, which encodes
the instantiation parameters (e.g., pose, texture) of the detected entity.

The "squashing" operation serves as the activation function for a capsule
layer. It ensures that the outputs from different capsules are on a
comparable scale before they are used in routing algorithms, such as
"dynamic routing." This decouples the "what" (existence probability, encoded
in the length) from the "how" (properties, encoded in the direction). Short
input vectors, representing low certainty, are shrunk almost to zero, while
long vectors, representing high certainty, are shrunk to a length just
below one.

Mathematical Foundation:
    The squashing function is defined as:
        ``v_squashed = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)``

    This formula can be understood as the product of two components:
    1.  **Directional Unit Vector (v / ||v||):** This term isolates the
        orientation of the input vector ``v``, ensuring that the direction of
        the output is identical to the input. This preserves the learned
        instantiation parameters.
    2.  **Scalar Scaling Factor (||v||^2 / (1 + ||v||^2)):** This term
        non-linearly scales the magnitude. It is a monotonic function of
        the squared norm ``||v||^2``.
        -   As the input vector's norm approaches zero (``||v|| -> 0``), the
            scaling factor also approaches zero, effectively nullifying
            low-confidence capsules.
        -   As the input vector's norm becomes very large (``||v|| -> inf``),
            the scaling factor asymptotically approaches one, ensuring the
            output length is always bounded.

References:
    - Sabour, S., Frosst, N., & Hinton, G. E. (2017). "Dynamic routing
      between capsules."
    - Hinton, G. E., Krizhevsky, A., & Wang, S. D. (2011). "Transforming
      auto-encoders." (Introduced the concept of capsules).

"""

import keras
from keras import ops, backend
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger


# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class SquashLayer(keras.layers.Layer):
    """Squashing non-linearity for Capsule Network vectors.

    Applies the squashing function
    ``squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)`` along a specified
    axis, normalizing capsule vector lengths to the range [0, 1) while
    preserving their directional information. Short vectors are shrunk toward
    zero and long vectors are shrunk to just below one.

    **Architecture Overview:**

    .. code-block:: text

        Input: v (batch, num_capsules, dim)
                │
                ├──────────────────────────────┐
                │                              │
                ▼                              ▼
        ┌───────────────────┐   ┌──────────────────────────┐
        │  Squared L2 Norm  │   │   Unit Vector:           │
        │  ||v||^2 along    │   │   v / (||v|| + epsilon)  │
        │  specified axis   │   │                          │
        └───────┬───────────┘   └─────────────┬────────────┘
                │                             │
                ▼                             │
        ┌───────────────────┐                 │
        │  Scale Factor:    │                 │
        │  ||v||^2          │                 │
        │  ───────────      │                 │
        │  1 + ||v||^2      │                 │
        └───────┬───────────┘                 │
                │                             │
                └──────────┬──────────────────┘
                           │
                           ▼
                ┌──────────────────┐
                │ scale * unit_vec │
                └────────┬─────────┘
                         │
                         ▼
        Output: (batch, num_capsules, dim)
                norms in [0, 1)

    :param axis: Axis along which to compute the vector norm for squashing.
        Defaults to -1 (last axis).
    :type axis: int
    :param epsilon: Small constant for numerical stability to prevent division
        by zero. If None, uses ``keras.backend.epsilon()`` (typically 1e-7).
    :type epsilon: Optional[float]
    :param kwargs: Additional keyword arguments passed to the Layer base class,
        such as ``name``, ``dtype``, ``trainable``, etc.

    References:
        - Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between
          capsules. In Advances in neural information processing systems.
        - Hinton, G. E., Krizhevsky, A., & Wang, S. D. (2011). Transforming
          auto-encoders. In International conference on artificial neural networks.
    """

    def __init__(
            self,
            axis: int = -1,
            epsilon: Optional[float] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the Squash layer.

        :param axis: Axis along which to compute vector norms for squashing.
        :type axis: int
        :param epsilon: Small constant for numerical stability. If None, uses
            ``keras.backend.epsilon()``.
        :type epsilon: Optional[float]
        :param kwargs: Additional keyword arguments for the Layer base class.
        """
        super().__init__(**kwargs)

        # Store configuration
        self.axis = axis
        self.epsilon = epsilon if epsilon is not None else backend.epsilon()

        logger.debug(f"Initialized SquashLayer with axis={axis}, epsilon={self.epsilon}")

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Apply squashing non-linearity to input vectors.

        Computes ``squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)``.

        :param inputs: Input tensor to be squashed. Vectors are identified along
            the specified axis.
        :type inputs: keras.KerasTensor
        :param training: Boolean indicating training or inference mode. Not used
            in this layer but kept for API consistency.
        :type training: Optional[bool]
        :return: Tensor with same shape as inputs, containing squashed vectors with
            norms bounded in [0, 1).
        :rtype: keras.KerasTensor
        """
        # Compute squared L2 norm along specified axis
        # Shape: input_shape with axis dimension reduced to 1
        squared_norm = ops.sum(
            ops.square(inputs),
            axis=self.axis,
            keepdims=True
        )

        # Compute safe norm to avoid division by zero
        # Add epsilon for numerical stability
        safe_norm = ops.sqrt(squared_norm + self.epsilon)

        # Compute scale factor: ||v||^2 / (1 + ||v||^2)
        # This ensures output norm is in [0, 1)
        scale = squared_norm / (1.0 + squared_norm)

        # Compute unit vector: v / ||v||
        unit_vector = inputs / safe_norm

        # Apply squashing: scale * unit_vector
        # Final result: (||v||^2 / (1 + ||v||^2)) * (v / ||v||)
        return scale * unit_vector

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple, identical to input_shape.
        :rtype: Tuple[Optional[int], ...]
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration for serialization.

        :return: Dictionary containing the layer configuration including
            axis and epsilon parameters along with parent class configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

    def __repr__(self) -> str:
        """Return string representation of the layer.

        :return: String representation including the layer name and key parameters.
        :rtype: str
        """
        return f"SquashLayer(axis={self.axis}, epsilon={self.epsilon}, name='{self.name}')"

# ---------------------------------------------------------------------
