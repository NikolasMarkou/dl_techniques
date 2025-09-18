"""
OrthoGLU Feed Forward Network Implementation
===========================================

This module implements an Orthogonally-Regularized Gated Linear Unit (OrthoGLU)
Feed Forward Network. This architecture synergizes the structured representation
learning of the `OrthoBlock` with the dynamic information routing of the GLU
mechanism.

This implementation strictly follows modern Keras 3 best practices for robust,
serializable, and production-ready custom layers.
"""

import keras
from keras import ops, layers, activations
from typing import Optional, Union, Any, Dict, Callable, Tuple

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from ..orthoblock import OrthoBlock

# ---------------------------------------------------------------------


@keras.saving.register_keras_serializable()
class OrthoGLUFFN(keras.layers.Layer):
    """
    Orthogonally-Regularized Gated Linear Unit Feed-Forward Network.

    This layer integrates the principles of the `OrthoBlock` into the Gated
    Linear Unit (GLU) architecture. It replaces standard Dense projections with
    `OrthoBlock` layers to enforce feature decorrelation and stability.

    **Intent (Doctrine)**: Implement "disciplined routing".
    1. **Discipline**: A first `OrthoBlock` projects the input into a
       structured, decorrelated, and stable high-dimensional space.
    2. **Routing**: This clean representation is then split into a `gate` and
       `value`. The GLU mechanism performs dynamic, content-aware selection.
    3. **Consolidation**: A second `OrthoBlock` projects the gated result to
       the final output dimension, re-imposing structural order.

    **Architecture**:
    ```
    Input(shape=[..., input_dim])
           ↓
    OrthoBlock(hidden_dim * 2) ──> Split into (gate, value)
           │                             │
           ├─> gate ─> Activation ──────┐
           │                             ▼
           └─> value ──────────────────> Multiply ──> Dropout ──> OrthoBlock
                                                                       │
                                                                       ▼
                                                           Output(shape=[...])
    ```

    **Data Flow**:
    1. Structured projection to expand dimensionality using `OrthoBlock`.
    2. Split into `gate` and `value` tensors of equal size.
    3. Apply activation function to the `gate` tensor.
    4. Element-wise multiplication of the activated gate with the `value`.
    5. Apply dropout for regularization.
    6. Final structured projection to the target output dimension.

    Args:
        hidden_dim: Integer, dimensionality of the intermediate hidden layer
            (size of `gate` and `value` tensors). Must be positive.
        output_dim: Integer, dimensionality of the final output. Must be
            positive.
        activation: Activation function for the gate. Can be a string name
            ('gelu', 'relu') or a callable. Defaults to 'gelu'.
        dropout_rate: Float between 0 and 1, dropout rate applied after
            gating for regularization. Defaults to 0.0.
        use_bias: Boolean, whether the internal `OrthoBlock` dense layers use
            bias. Defaults to True.
        ortho_reg_factor: Float or Tuple[float, float]. Regularization
            strength for the input and output `OrthoBlock`s. If a single
            float is provided, it is used for both. Defaults to 0.01.
        scale_initial_value: Float or Tuple[float, float]. Initial value for
            the constrained scaling gates inside the `OrthoBlock`s.
            Defaults to 0.9.
        **kwargs: Additional keyword arguments for the base Layer class.

    Input shape:
        N-D tensor with shape: `(batch_size, ..., input_dim)`.

    Output shape:
        N-D tensor with shape: `(batch_size, ..., output_dim)`.

    Attributes:
        input_proj_ortho: The first `OrthoBlock` for input projection.
        output_proj_ortho: The second `OrthoBlock` for output projection.
        dropout: Dropout layer for regularization.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        ortho_reg_factor: Union[float, Tuple[float, float]] = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        # 1. Comprehensive Input Validation
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(
                f"dropout_rate must be between 0 and 1, got {dropout_rate}"
            )

        # 2. Store ALL Configuration Parameters for serialization
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.ortho_reg_factor = ortho_reg_factor

        # Unpack tuple or duplicate single value for block-specific configs
        ortho_factors = (
            ortho_reg_factor
            if isinstance(ortho_reg_factor, tuple)
            else (ortho_reg_factor, ortho_reg_factor)
        )

        # 3. CREATE all sub-layers in __init__ (The Golden Rule)
        self.input_proj_ortho = OrthoBlock(
            units=hidden_dim * 2,
            activation=None,
            use_bias=False,
            ortho_reg_factor=ortho_factors[0],
            name="input_proj_ortho",
        )

        self.output_proj_ortho = OrthoBlock(
            units=output_dim,
            activation=None,
            use_bias=self.use_bias,
            ortho_reg_factor=ortho_factors[1],
            name="output_proj_ortho",
        )

        self.dropout = layers.Dropout(dropout_rate, name="dropout")

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        CRITICAL: For composite layers, you MUST explicitly build each
        sub-layer. This ensures their weight variables are created before
        Keras attempts to load saved weights during deserialization.
        """
        if input_shape[-1] is None:
            raise ValueError("Last dimension of input must be defined")

        # Build sub-layers in computational order
        self.input_proj_ortho.build(input_shape)

        intermediate_shape = (*input_shape[:-1], self.hidden_dim)
        self.dropout.build(intermediate_shape)
        self.output_proj_ortho.build(intermediate_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Forward pass for the OrthoGLU FFN."""
        gate_and_value = self.input_proj_ortho(inputs, training=training)

        gate, value = ops.split(gate_and_value, indices_or_sections=2, axis=-1)
        activated_gate = self.activation(gate)
        gated_value = activated_gate * value

        gated_value = self.dropout(gated_value, training=training)

        output = self.output_proj_ortho(gated_value, training=training)

        return output

    def compute_output_shape(
        self, input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Computes the output shape of the layer."""
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the layer's configuration for serialization.

        CRITICAL: Must include ALL parameters from __init__ for proper
        reconstruction.
        """
        config = super().get_config()
        config.update(
            {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "activation": activations.serialize(self.activation),
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "ortho_reg_factor": self.ortho_reg_factor
            }
        )
        return config

# ---------------------------------------------------------------------
