import keras
from enum import Enum
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, Union

# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class LayerScale(keras.layers.Layer):
    """
    Layer for learning per-channel scaling factors.

    This layer multiplies each channel of the input by a learnable scalar.
    Useful for stabilizing training in deep networks.

    Args:
        init_values (float): Initial value for the scaling factors.
        projection_dim (int): Number of channels/dimensions to scale.
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            init_values: float,
            projection_dim: int,
            **kwargs: Any
    ) -> None:
        """Initialize the LayerScale layer."""
        super().__init__(**kwargs)
        self.init_values = init_values
        self.projection_dim = projection_dim
        self.gamma = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer.

        Args:
            input_shape: Shape of input tensor.
        """
        self.gamma = self.add_weight(
            name="gamma",
            shape=(self.projection_dim,),
            initializer=keras.initializers.Constant(self.init_values),
            trainable=True,
            dtype=self.dtype
        )

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> tf.Tensor:
        """
        Apply layer scaling to inputs.

        Args:
            inputs: Input tensor of shape [..., projection_dim].
            training: Whether in training mode (unused).
            **kwargs: Additional call arguments.

        Returns:
            Scaled tensor of same shape as input.
        """
        return inputs * self.gamma

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "init_values": self.init_values,
            "projection_dim": self.projection_dim,
        })
        return config

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape."""
        return input_shape


# ---------------------------------------------------------------------

class MultiplierType(Enum):
    GLOBAL = 0

    CHANNEL = 1

    @staticmethod
    def from_string(type_str: Union[str, "MultiplierType"]) -> "MultiplierType":
        # --- argument checking
        if type_str is None:
            raise ValueError("type_str must not be null")
        if isinstance(type_str, MultiplierType):
            return type_str
        if not isinstance(type_str, str):
            raise ValueError("type_str must be string")
        # --- clean string and get
        type_str = type_str.strip().upper()
        if len(type_str) <= 0:
            raise ValueError("stripped type_str must not be empty")
        return MultiplierType[type_str]

    def to_string(self) -> str:
        return self.name


# ---------------------------------------------------------------------


@keras.utils.register_keras_serializable()
class LearnableMultiplier(keras.layers.Layer):
    """
    Layer implementing learnable multipliers.

    The multipliers can be either global (single value) or per-channel.
    Values are initialized close to 1.0 and incentivised to stay near 1.0.

    Args:
        multiplier_type: Type of multiplier ('GLOBAL' or 'CHANNEL').
        capped: Whether to cap multiplier values (default: True).
        initializer: Weight initializer (default: truncated normal).
        regularizer: Weight regularizer (default: L2).
        **kwargs: Additional layer arguments.
    """

    def __init__(
            self,
            multiplier_type: Union[MultiplierType, str],
            capped: bool = True,
            initializer: Optional[keras.initializers.Initializer] = None,
            regularizer: Optional[keras.regularizers.Regularizer] = None,
            **kwargs: Any
    ) -> None:
        """Initialize the LearnableMultiplier layer."""
        super().__init__(**kwargs)

        # Handle default initializer
        if initializer is None:
            initializer = keras.initializers.TruncatedNormal(
                mean=0.0,
                stddev=0.01,
                seed=0
            )

        # Handle default regularizer
        if regularizer is None:
            regularizer = keras.regularizers.L2(l2=1e-2)

        self.capped = capped
        self.initializer = initializer
        self.regularizer = regularizer
        self.multiplier_type = MultiplierType.from_string(multiplier_type)
        self.w_multiplier = None

    def build(self, input_shape: tf.TensorShape) -> None:
        """
        Build the layer by creating the multiplier weights.

        Args:
            input_shape: Shape of input tensor.
        """
        # Create shape with ones except for channel dimension if needed
        if self.multiplier_type == MultiplierType.GLOBAL:
            weight_shape = [1] * len(input_shape)
        elif self.multiplier_type == MultiplierType.CHANNEL:
            weight_shape = [1] * (len(input_shape) - 1) + [input_shape[-1]]

        self.w_multiplier = self.add_weight(
            name="w_multiplier",
            shape=weight_shape,
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True,
            dtype=self.dtype
        )

    def call(
            self,
            inputs: tf.Tensor,
            training: Optional[bool] = None,
            **kwargs: Any
    ) -> tf.Tensor:
        """
        Apply the learnable multipliers to inputs.

        The multipliers are transformed using tanh to constrain their values.
        When capped=True, values are constrained to [0, 1].
        When capped=False, values can exceed 1.0.

        Args:
            inputs: Input tensor.
            training: Whether in training mode (unused).
            **kwargs: Additional call arguments.

        Returns:
            Tensor with multipliers applied.
        """
        # Compute base multiplier using tanh transformation
        base_multiplier = tf.nn.tanh((4 * self.w_multiplier + 1.5) + 1) * 0.5

        if self.capped:
            multiplier = base_multiplier
        else:
            # Allow values > 1.0 when uncapped
            multiplier = base_multiplier * (1.0 + self.w_multiplier)

        return tf.multiply(multiplier, inputs)

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            "multiplier_type": self.multiplier_type.to_string(),
            "capped": self.capped,
            "regularizer": keras.regularizers.serialize(self.regularizer),
            "initializer": keras.initializers.serialize(self.initializer)
        })
        return config

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Compute output shape (same as input shape)."""
        return input_shape

# ---------------------------------------------------------------------
