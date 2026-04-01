"""
A dual-pathway feed-forward network for differential processing.

This layer introduces a non-standard feed-forward architecture inspired by
the principle of push-pull or opponent processing found in biological neural
systems, such as the human visual system. Instead of processing input signals
through a single, monolithic pathway, it first decomposes the input into its
positive and negative components and processes them through two parallel,
specialized sub-networks. The final representation is derived from the
difference between the outputs of these two pathways.

The core hypothesis is that dedicating separate pathways for excitatory
(positive) and inhibitory (negative) signals allows the network to learn
more disentangled and robust features. This explicit separation forces the
model to learn what aspects of the input should "push" the representation
in a certain direction versus what should "pull" it away, potentially
leading to better gradient flow and more nuanced function approximation.

Architectural Overview:
The layer's data flow is structured as follows:

1.  **Input Decomposition**: The input tensor `x` is split into two
    non-negative tensors: a positive part `x_pos = ReLU(x)` and a negative
    part `x_neg = ReLU(-x)`. This ensures that for any given feature, its
    signal is active in only one of the two pathways.

2.  **Parallel Pathway Processing**: `x_pos` and `x_neg` are fed into two
    structurally identical but independently parameterized branches. Each
    branch consists of a sequence of linear transformations, layer
    normalization, and non-linear activations. This allows each pathway to
    learn specialized transformations for its respective signal type.

3.  **Differential Combination**: The outputs of the positive and negative
    pathways are combined through subtraction. This "differential" tensor
    represents the net effect, or the balance of evidence, between the
    features learned by the two opposing branches.

4.  **Output Projection**: The resulting differential tensor is further
    normalized and passed through a final linear projection to produce the
    layer's output.

Foundational Mathematics:
Let `x` be the input vector. The layer's computation can be expressed as:

1.  Input Splitting:
    `x_pos = max(0, x)`
    `x_neg = max(0, -x)`

2.  Branch Functions: Let `f_pos(.)` and `f_neg(.)` represent the learned
    functions of the positive and negative pathways, respectively. Each
    function is a composition of Dense, LayerNorm, and Activation layers.
    `h_pos = f_pos(x_pos)`
    `h_neg = f_neg(x_neg)`

3.  Differential Computation: The core operation is the subtraction of the
    pathway outputs.
    `h_diff = h_pos - h_neg`

4.  Final Output: The differential representation is then transformed to the
    final output dimension.
    `y = W_out @ h_diff + b_out`

This architecture transforms the standard FFN computation `y = f(x)` into
`y = g(f_pos(max(0, x)) - f_neg(max(0, -x)))`, forcing the model to learn a
function as a difference of two non-negative component functions. The default
use of a `SoftOrthonormalConstraintRegularizer` on the weights encourages
the learned transformations within each pathway to be well-conditioned,
preventing feature collapse and promoting stable training dynamics.

References:
The design synthesizes several key ideas in modern deep learning:

-   The input splitting is a core component of the Concatenated ReLU (CReLU)
    activation, which was proposed to preserve information by handling
    positive and negative phases separately.
    - Shang, W., et al. (2016). Understanding and Improving Convolutional
      Neural Networks via Concatenated Rectified Linear Units. ICML.

-   The use of Layer Normalization is critical for stabilizing the activations
    in each independent pathway, a technique introduced in:
    - Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization.
      arXiv preprint arXiv:1607.06450.

-   The concept of opponent processing is a foundational principle in
    neuroscience, particularly in models of sensory perception.

"""

import keras
from typing import Callable, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class DifferentialFFN(keras.layers.Layer):
    """
    Differential Feed-Forward Network with dual-pathway processing.

    This layer decomposes input into positive (``x_pos = ReLU(x)``) and negative
    (``x_neg = ReLU(-x)``) components, processes each through independent
    Dense-LayerNorm-Activation-Gate branches, computes their difference
    ``h_diff = f_pos(x_pos) - f_neg(x_neg)``, and projects the result through
    LayerNorm, Dropout, and a final Dense layer to produce the output. This
    biologically-inspired opponent processing enables more disentangled feature
    learning and improved gradient flow.

    **Architecture Overview:**

    .. code-block:: text

        ┌──────────────────────────────────┐
        │     Input (..., input_dim)        │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │  Split: ReLU(x)  │  ReLU(-x)     │
        └───────┬──────────┴───────┬───────┘
                ▼                  ▼
        ┌──────────────┐   ┌──────────────┐
        │ Positive Path │   │ Negative Path │
        │ Dense(hidden) │   │ Dense(hidden) │
        │  LayerNorm    │   │  LayerNorm    │
        │  Activation   │   │  Activation   │
        │ Dense(hidden/2)│  │ Dense(hidden/2)│
        │  Gate Activ.  │   │  Gate Activ.  │
        └───────┬──────┘   └───────┬──────┘
                │                  │
                └────────┬─────────┘
                         ▼
        ┌──────────────────────────────────┐
        │   Differential: pos - neg         │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │          LayerNorm                │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │       Dropout (optional)          │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │   output_proj: Dense(output_dim)  │
        └──────────────┬───────────────────┘
                       ▼
        ┌──────────────────────────────────┐
        │     Output (..., output_dim)      │
        └──────────────────────────────────┘

    :param hidden_dim: Integer, dimension of the hidden layer in each branch. Must be positive
        and divisible by 2 for proper gating projection.
    :type hidden_dim: int
    :param output_dim: Integer, dimension of the output. Must be positive.
    :type output_dim: int
    :param branch_activation: Activation function used in the branches.
        Accepts standard activation names ('gelu', 'relu', 'swish') or callables.
        Defaults to 'gelu'.
    :type branch_activation: Union[str, Callable]
    :param gate_activation: Activation function used in the gate projections.
        Typically 'sigmoid' for proper gating behavior. Defaults to 'sigmoid'.
    :type gate_activation: Union[str, Callable]
    :param dropout_rate: Float between 0.0 and 1.0, dropout rate applied to differential features.
        Provides regularization to prevent overfitting. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether to use bias terms in dense layers. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for kernel weights.
        Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for bias weights.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for kernel weights.
        If None, uses SoftOrthonormalConstraintRegularizer for stability.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Optional regularizer for bias weights.
        Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If ``hidden_dim`` is not positive or not divisible by 2.
    :raises ValueError: If ``output_dim`` is not positive.
    :raises ValueError: If ``dropout_rate`` is not between 0.0 and 1.0.

    Note:
        The ``hidden_dim`` must be divisible by 2 because each branch's gating projection
        maps to ``hidden_dim // 2`` dimensions, ensuring the final differential features
        have consistent dimensionality.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        branch_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        gate_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "sigmoid",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, keras.initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, keras.initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be divisible by 2, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0.0 and 1.0, got {dropout_rate}")

        # Store configuration
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.branch_activation = keras.activations.get(branch_activation)
        self.gate_activation = keras.activations.get(gate_activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

        # Handle regularizer - use default if None provided
        if kernel_regularizer is None:
            self.kernel_regularizer = SoftOrthonormalConstraintRegularizer()
        else:
            self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # CREATE all sub-layers in __init__ (Pattern 2: Composite Layer)
        # Following modern Keras 3 pattern - create but don't build here

        # Positive branch: Dense -> LayerNorm -> Activation -> Dense(gate)
        self.positive_dense = keras.layers.Dense(
            units=self.hidden_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="positive_dense"
        )
        self.layer_norm_pos = keras.layers.LayerNormalization(
            center=True,
            scale=True,
            name="layer_norm_positive"
        )
        self.positive_proj = keras.layers.Dense(
            units=self.hidden_dim // 2,
            activation=None,  # Activation applied separately for clarity
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="positive_proj"
        )

        # Negative branch: Dense -> LayerNorm -> Activation -> Dense(gate)
        self.negative_dense = keras.layers.Dense(
            units=self.hidden_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="negative_dense"
        )
        self.layer_norm_neg = keras.layers.LayerNormalization(
            center=True,
            scale=True,
            name="layer_norm_negative"
        )
        self.negative_proj = keras.layers.Dense(
            units=self.hidden_dim // 2,
            activation=None,  # Activation applied separately for clarity
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="negative_proj"
        )

        # Differential processing layers
        self.layer_norm_diff = keras.layers.LayerNormalization(
            center=False,  # No centering for differential features
            scale=True,
            name="layer_norm_diff"
        )
        self.dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )
        self.output_proj = keras.layers.Dense(
            units=self.output_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="output_proj"
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        Explicitly builds each sub-layer for robust serialization following
        the modern Keras 3 composite layer pattern.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        """
        # Build positive branch sub-layers in computational order
        self.positive_dense.build(input_shape)
        dense_output_shape = self.positive_dense.compute_output_shape(input_shape)
        self.layer_norm_pos.build(dense_output_shape)
        self.positive_proj.build(dense_output_shape)

        # Build negative branch sub-layers
        self.negative_dense.build(input_shape)
        # Note: negative branch has same architecture as positive
        self.layer_norm_neg.build(dense_output_shape)
        self.negative_proj.build(dense_output_shape)

        # Build differential processing layers
        proj_output_shape = self.positive_proj.compute_output_shape(dense_output_shape)
        self.layer_norm_diff.build(proj_output_shape)
        self.dropout.build(proj_output_shape)
        self.output_proj.build(proj_output_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Differential FFN layer.

        :param inputs: Input tensor with shape (..., input_dim).
        :type inputs: keras.KerasTensor
        :param training: Whether the layer should behave in training mode
            (applies dropout) or inference mode.
        :type training: Optional[bool]
        :return: Output tensor with shape (..., output_dim).
        :rtype: keras.KerasTensor
        """
        # Split input into positive and negative components
        inputs_positive = keras.ops.relu(inputs)
        inputs_negative = keras.ops.relu(-inputs)

        # Positive branch processing
        pos_hidden = self.positive_dense(inputs_positive)
        pos_normed = self.layer_norm_pos(pos_hidden, training=training)
        pos_activated = self.branch_activation(pos_normed)
        pos_projected = self.positive_proj(pos_activated)
        pos_gated = self.gate_activation(pos_projected)

        # Negative branch processing
        neg_hidden = self.negative_dense(inputs_negative)
        neg_normed = self.layer_norm_neg(neg_hidden, training=training)
        neg_activated = self.branch_activation(neg_normed)
        neg_projected = self.negative_proj(neg_activated)
        neg_gated = self.gate_activation(neg_projected)

        # Compute differential representation
        differential = pos_gated - neg_gated

        # Process differential features
        diff_normed = self.layer_norm_diff(differential, training=training)
        diff_dropped = self.dropout(diff_normed, training=training)

        # Final projection to output dimension
        output = self.output_proj(diff_dropped)

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape tuple with last dimension as output_dim.
        :rtype: Tuple[Optional[int], ...]
        """
        output_shape = list(input_shape)
        output_shape[-1] = self.output_dim
        return tuple(output_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        :return: Dictionary containing the layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'branch_activation': keras.activations.serialize(self.branch_activation),
            'gate_activation': keras.activations.serialize(self.gate_activation),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
