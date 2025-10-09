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
        v_squashed = (||v||² / (1 + ||v||²)) * (v / ||v||)

    This formula can be understood as the product of two components:
    1.  **Directional Unit Vector (`v / ||v||`):** This term isolates the
        orientation of the input vector `v`, ensuring that the direction of
        the output is identical to the input. This preserves the learned
        instantiation parameters.
    2.  **Scalar Scaling Factor (`||v||² / (1 + ||v||²)`):** This term
        non-linearly scales the magnitude. It is a monotonic function of
        the squared norm `||v||²`.
        -   As the input vector's norm approaches zero (`||v|| -> 0`), the
            scaling factor also approaches zero, effectively nullifying
            low-confidence capsules.
        -   As the input vector's norm becomes very large (`||v|| -> ∞`),
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
    """Applies squashing non-linearity to vectors (capsules).

    The squashing function is a key component in Capsule Networks that ensures
    capsule outputs have meaningful magnitudes while preserving their directional
    information. This function provides the following properties:

    1. Short vectors get shrunk to almost zero length
    2. Long vectors get shrunk to a length slightly below 1  
    3. Vector orientation is preserved throughout the transformation

    Mathematical formulation:
        squash(v) = (||v||² / (1 + ||v||²)) * (v / ||v||)

    Where ||v|| represents the L2 norm of vector v, and the operation is applied
    along the specified axis. The resulting vectors have norms in the range [0, 1).

    Args:
        axis: Integer, axis along which to compute the vector norm for squashing.
            Defaults to -1 (last axis). This determines which dimension represents
            the vector components to be squashed.
        epsilon: Float, small constant for numerical stability to prevent division
            by zero. If None, uses keras.backend.epsilon() which is typically 1e-7.
            Defaults to None.
        **kwargs: Additional keyword arguments passed to the Layer base class,
            such as `name`, `dtype`, `trainable`, etc.

    Input shape:
        Arbitrary tensor of rank >= 1. The squashing operation is applied along
        the specified axis, treating slices along that axis as vectors.

    Output shape:
        Same as input shape. No dimensional transformation occurs.

    Attributes:
        axis: The axis along which vector norms are computed.
        epsilon: The small constant used for numerical stability.

    Example:
        ```python
        # Basic usage for capsule vectors
        layer = SquashLayer()
        inputs = keras.Input(shape=(10, 16))  # 10 capsules, each 16-dimensional
        outputs = layer(inputs)  # Same shape, but vectors are squashed

        # Custom axis for different tensor layouts
        layer = SquashLayer(axis=1)
        inputs = keras.Input(shape=(32, 10, 16))  # Batch of capsule matrices
        outputs = layer(inputs)  # Squashing applied along axis 1

        # In a Capsule Network
        inputs = keras.Input(shape=(1152, 8))  # Primary capsules
        x = keras.layers.Dense(160)(inputs)  # Transform to digit capsules
        x = keras.layers.Reshape((10, 16))(x)  # 10 digit capsules, 16D each
        outputs = SquashLayer()(x)  # Apply squashing to capsule vectors

        # Custom epsilon for numerical stability
        layer = SquashLayer(epsilon=1e-8)
        ```

    References:
        - Sabour, S., Frosst, N., & Hinton, G. E. (2017). Dynamic routing between
          capsules. In Advances in neural information processing systems.
        - Hinton, G. E., Krizhevsky, A., & Wang, S. D. (2011). Transforming
          auto-encoders. In International conference on artificial neural networks.

    Note:
        - The squashing function is non-linear and differentiable
        - Output vector norms are bounded in the range [0, 1)
        - This layer has no trainable parameters
        - Commonly used as the final activation in capsule layers
        - The epsilon parameter is crucial for numerical stability when vectors
          have very small norms
    """

    def __init__(
            self,
            axis: int = -1,
            epsilon: Optional[float] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the Squash layer.

        Args:
            axis: Axis along which to compute vector norms for squashing.
            epsilon: Small constant for numerical stability. If None, uses
                keras.backend.epsilon().
            **kwargs: Additional keyword arguments for the Layer base class.
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

        Applies the squashing function: squash(v) = (||v||² / (1 + ||v||²)) * (v / ||v||)

        Args:
            inputs: Input tensor to be squashed. Vectors are identified along
                the specified axis.
            training: Boolean indicating whether the layer should behave in
                training mode or inference mode. Not used in this layer but
                kept for API consistency.

        Returns:
            Tensor with same shape as inputs, containing squashed vectors with
            norms bounded in [0, 1).
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

        # Compute scale factor: ||v||² / (1 + ||v||²)
        # This ensures output norm is in [0, 1)
        scale = squared_norm / (1.0 + squared_norm)

        # Compute unit vector: v / ||v||
        unit_vector = inputs / safe_norm

        # Apply squashing: scale * unit_vector
        # Final result: (||v||² / (1 + ||v||²)) * (v / ||v||)
        return scale * unit_vector

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Compute the output shape of the layer.

        For the squash layer, output shape is identical to input shape since
        no dimensional transformation occurs - only the vector magnitudes change.

        Args:
            input_shape: Shape tuple of the input tensor.

        Returns:
            Output shape tuple, identical to input_shape.
        """
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration for serialization.

        Returns all parameters passed to __init__ so the layer can be
        properly reconstructed during model loading.

        Returns:
            Dictionary containing the layer configuration, including
            axis and epsilon parameters along with parent class configuration.
        """
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

    def __repr__(self) -> str:
        """Return string representation of the layer.

        Returns:
            String representation including the layer name and key parameters.
        """
        return f"SquashLayer(axis={self.axis}, epsilon={self.epsilon}, name='{self.name}')"

# ---------------------------------------------------------------------