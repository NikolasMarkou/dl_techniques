"""
An orthogonally-regularized Gated Linear Unit FFN.

This layer synergizes two powerful concepts: the dynamic, input-dependent
information routing of Gated Linear Units (GLU) and the training stability
and feature decorrelation benefits of orthogonal transformations. By replacing
the standard dense projections in a GLU with orthogonally-regularized blocks,
this layer aims to perform "disciplined routing," where feature selection
occurs within a structured, well-behaved latent space.

The core design philosophy is to first map the input into a stable and
information-rich representation and then apply the selective gating
mechanism. This prevents the feature collapse and gradient instability that
can sometimes affect deep or complex feed-forward networks.

Architectural Overview:
The layer's architecture enforces structure at both its input and output:

1.  **Structured Projection**: The input tensor is first projected into a
    higher-dimensional space using an `OrthoBlock`. This block's internal
    weight matrix is regularized towards orthogonality. This initial step
    transforms the input features into a decorrelated basis, ensuring a
    rich and stable representation for subsequent processing.

2.  **Gated Routing**: The orthogonally projected tensor is split into two
    halves: a "gate" and a "value." The gate is passed through a non-linear
    activation function and then element-wise multiplied with the value. This
    is the standard GLU mechanism, which dynamically filters the information
    in the value tensor based on the input-dependent gate.

3.  **Structured Consolidation**: The resulting gated tensor is projected back
    to the final output dimension using a second `OrthoBlock`. This step
    re-imposes structural discipline, ensuring the final output
    representation also inhabits a stable, decorrelated space.

Foundational Mathematics:
The layer's computation combines the GLU formulation with the properties of
orthogonal matrices. Let `x` be the input vector.

1.  The gated representation `h` is formed as:
    `[g', v] = O_in(x)`
    `h = activation(g') * v`
    where `O_in` represents the input `OrthoBlock` transformation and `*` is
    the element-wise product.

2.  The key component is the `OrthoBlock`, which implements a linear
    transformation `y = Wx` where the weight matrix `W` is encouraged to be
    orthogonal. An orthogonal matrix satisfies the property `W^T W = I`,
    where `I` is the identity matrix. This property has profound implications:
    -   **Norm Preservation**: Orthogonal transformations are isometries,
        meaning they preserve vector norms (`||Wx||_2 = ||x||_2`). This helps
        prevent the magnitude of activations from exploding or vanishing,
        leading to more stable training.
    -   **Gradient Stability**: During backpropagation, the gradient norm is
        also preserved (`||∇_x L|| = ||W^T ∇_y L|| = ||∇_y L||`). This ensures
        that error signals propagate effectively through the layer without
        being unduly scaled, mitigating the vanishing and exploding gradient
        problems.
    -   **Feature Decorrelation**: Orthogonality regularization encourages
        the rows (and columns) of the weight matrix to be orthonormal. This
        forces the layer to learn a set of non-redundant filters, maximizing
        the information capacity of the learned representation.

The synergy of these components allows the `OrthoGLUFFN` to learn a highly
expressive, input-adaptive function while maintaining the desirable training
dynamics associated with orthogonal transformations.

References:
The GLU mechanism and its variants are detailed in:
-   Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv preprint
    arXiv:2002.05202.

The benefits and techniques of using orthogonality in deep learning are
explored in:
-   Bansal, N., et al. (2018). Can We Gain More from Orthogonality
    Regularizations in Training Deep Networks? NeurIPS.
-   Cisse, M., et al. (2017). Parseval Networks: Improving Robustness to
    Adversarial Examples. ICML.

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
