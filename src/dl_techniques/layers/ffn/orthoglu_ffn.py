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

    This layer integrates the principles of the ``OrthoBlock`` into the Gated
    Linear Unit (GLU) architecture, replacing standard Dense projections with
    orthogonally-regularized blocks to enforce feature decorrelation and stability.
    The computation is ``output = O_out(activation(gate) * value)`` where
    ``[gate, value] = split(O_in(x))`` and ``O_in``, ``O_out`` are orthogonally-
    regularized linear transformations satisfying ``W^T W ≈ I``.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │     Input (..., input_dim)       │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │ OrthoBlock(hidden_dim * 2)       │
        │ (orthogonal regularization)      │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │    Split into (gate, value)      │
        └───────┬──────────────────┬───────┘
                ▼                  │
        ┌──────────────┐           │
        │  Activation  │           │
        └───────┬──────┘           │
                │                  │
                └──────────┬───────┘
                           ▼
        ┌──────────────────────────────────┐
        │      Element-wise Multiply       │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │       Dropout (optional)         │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │    OrthoBlock(output_dim)        │
        │    (orthogonal regularization)   │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │    Output (..., output_dim)      │
        └──────────────────────────────────┘

    :param hidden_dim: Integer, dimensionality of the intermediate hidden layer
        (size of gate and value tensors). Must be positive.
    :type hidden_dim: int
    :param output_dim: Integer, dimensionality of the final output. Must be
        positive.
    :type output_dim: int
    :param activation: Activation function for the gate. Can be a string name
        ('gelu', 'relu') or a callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Float between 0 and 1, dropout rate applied after
        gating for regularization. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the internal OrthoBlock dense layers use
        bias. Defaults to True.
    :type use_bias: bool
    :param ortho_reg_factor: Float or Tuple[float, float]. Regularization
        strength for the input and output OrthoBlocks. If a single
        float is provided, it is used for both. Defaults to 1.0.
    :type ortho_reg_factor: Union[float, Tuple[float, float]]
    :param kwargs: Additional keyword arguments for the base Layer class.
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

        Explicitly builds each sub-layer to ensure their weight variables are
        created before Keras attempts to load saved weights during deserialization.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
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
        """
        Forward pass for the OrthoGLU FFN.

        :param inputs: Input tensor of shape (..., input_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode.
        :type training: Optional[bool]
        :return: Output tensor of shape (..., output_dim).
        :rtype: keras.KerasTensor
        """
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
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with last dimension = output_dim.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the layer's configuration for serialization.

        :return: Dictionary containing all parameters from __init__ for proper
            reconstruction.
        :rtype: Dict[str, Any]
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
