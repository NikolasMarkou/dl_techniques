"""
A dual-pathway feed-forward network that subtracts two branches.

This layer splits its input into a positive part and a negative part, runs
each through its own branch, and subtracts one branch's output from the
other. The idea comes from push-pull, or opponent, processing in biological
sensory systems: one channel says "more", the other says "less", and what
matters is the difference.

The hypothesis is that giving the excitatory and the inhibitory signal their
own weights produces more disentangled features than one shared pathway.

**Architecture Overview:**

1.  **Input decomposition**. The input ``x`` becomes two non-negative
    tensors: ``x_pos = ReLU(x)`` and ``x_neg = ReLU(-x)``. Every feature is
    active in at most one of the two, so the branches see disjoint signals.
    At ``x == 0`` neither is active.

2.  **Parallel branches**. ``x_pos`` and ``x_neg`` go through two branches with
    the same structure and separate weights: Dense to ``hidden_dim``,
    LayerNorm, ``branch_activation``, Dense to ``hidden_dim // 2``, then
    ``gate_activation``.

3.  **Difference**. The two branch outputs are subtracted. This is the net
    evidence: what the positive branch found minus what the negative branch
    found.

4.  **Output projection**. The difference is normalized, passed through
    dropout, and projected to ``output_dim``.

**Mathematics:**
Let ``x`` be the input vector.

1.  Input splitting:
    ``x_pos = max(0, x)``
    ``x_neg = max(0, -x)``

2.  Branch functions. ``f_pos`` and ``f_neg`` are the learned branch
    functions, each a Dense, a LayerNorm, ``branch_activation``, a second
    Dense and ``gate_activation``:
    ``h_pos = f_pos(x_pos)``
    ``h_neg = f_neg(x_neg)``

3.  Difference:
    ``h_diff = h_pos - h_neg``

4.  Output. ``h_diff`` is normalized (with no centering), then dropped out,
    then projected:
    ``y = W_out @ dropout(norm(h_diff)) + b_out``

So the standard FFN form ``y = f(x)`` becomes
``y = g(f_pos(max(0, x)) - f_neg(max(0, -x)))``: the model has to express its
function as a difference of two non-negative component functions. When you
pass no ``kernel_regularizer``, the layer installs a
``SoftOrthonormalConstraintRegularizer``, which keeps each branch's
transformation well conditioned and stops the features collapsing.

References:
The design combines several existing ideas:

-   The input split is the core of the Concatenated ReLU (CReLU) activation,
    proposed to preserve information by handling the positive and negative
    phases separately.
    - Shang, W., et al. (2016). Understanding and Improving Convolutional
      Neural Networks via Concatenated Rectified Linear Units. ICML.

-   Layer Normalization stabilizes the activations inside each branch.
    - Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization.
      arXiv preprint arXiv:1607.06450.

-   Opponent processing is a foundational principle in neuroscience, in
    particular in models of sensory perception.

"""

import keras
from typing import Callable, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.regularizers.soft_orthogonal import SoftOrthonormalConstraintRegularizer
from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.ffn.diff_ffn")
class DifferentialFFN(keras.layers.Layer):
    """
    Differential feed-forward network with two subtracted pathways.

    The input is split into ``x_pos = ReLU(x)`` and ``x_neg = ReLU(-x)``. Each
    half runs through its own Dense - LayerNorm - activation - Dense -
    activation branch. The two branch outputs are subtracted, and the
    difference is normalized, dropped out and projected to ``output_dim``:
    ``y = output_proj(dropout(norm(f_pos(x_pos) - f_neg(x_neg))))``.

    Both branches end at ``hidden_dim // 2``, which is why ``hidden_dim`` has
    to be even. The input width is independent of ``hidden_dim`` and of
    ``output_dim``.

    **Architecture Overview:**

    .. code-block:: text

            Input  [..., input_dim]
                     │
               ┌─────┴─────┐
               ▼           ▼
            ReLU(x)     ReLU(-x)
               │           │
               ▼           ▼
        ┌────────────┐ ┌────────────┐
        │  positive  │ │  negative  │
        │   branch   │ │   branch   │
        └─────┬──────┘ └─────┬──────┘
              │ [..., D/2]   │ [..., D/2]
              └──────┬───────┘
                     ▼
             subtract: pos - neg  [..., D/2]
                     │
                     ▼
          ┌──────────────────────┐
          │   layer_norm_diff    │
          │    center=False      │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │       dropout        │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │     output_proj      │
          │  Dense(output_dim)   │
          └──────────┬───────────┘
                     ▼
            Output [..., output_dim]

        D = hidden_dim. The Dropout layer is ALWAYS created, at
        rate dropout_rate; at the 0.0 default it is a no-op that
        still sits in the graph. layer_norm_diff does not centre,
        because a difference is already centred at zero.

    **Branch internals (the two pathways):**

    .. code-block:: text

        x_pos = ReLU(x)               x_neg = ReLU(-x)
            │                             │
            ▼                             ▼
        positive_dense                negative_dense
        Dense(D)  [..., D]            Dense(D)  [..., D]
            │                             │
            ▼                             ▼
        layer_norm_pos                layer_norm_neg
            │                             │
            ▼                             ▼
        branch_activation             branch_activation
            │                             │
            ▼                             ▼
        positive_proj                 negative_proj
        Dense(D/2)                    Dense(D/2)
            │                             │
            ▼                             ▼
        gate_activation               gate_activation
            │                             │
            └───────► pos - neg ◄─────────┘
                      [..., D/2]

        Two activations at two places. branch_activation ('gelu'
        by default) runs after the LayerNorm. gate_activation
        ('sigmoid' by default) runs on the second Dense, and its
        output is what gets subtracted.

        Nothing is multiplied here. The "gate" is a squashing
        function, and the two paths are joined by subtraction. With
        the sigmoid default each branch output lies in (0, 1), so
        the difference lies in (-1, 1) before layer_norm_diff.

        The branches never share weights: every Dense gets its own
        clone of the initializer.

    :param hidden_dim: Width of the first Dense in each branch. Must be
        positive and even, because the second Dense halves it.
    :type hidden_dim: int
    :param output_dim: Width of the output. Must be positive.
    :type output_dim: int
    :param branch_activation: Activation applied after the LayerNorm in each
        branch. A name ('gelu', 'relu', 'swish') or a callable. Defaults to
        'gelu'.
    :type branch_activation: Union[str, Callable]
    :param gate_activation: Activation applied to the second Dense of each
        branch. 'sigmoid' bounds each branch to (0, 1). Defaults to 'sigmoid'.
    :type gate_activation: Union[str, Callable]
    :param dropout_rate: Dropout rate on the differential features, in
        ``[0.0, 1.0]``. Defaults to 0.0. The Dropout layer exists either way.
    :type dropout_rate: float
    :param use_bias: Whether the Dense layers carry a bias. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for the kernels. Each Dense gets a
        clone, never the same instance. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases, also cloned per
        Dense. Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. When None, the
        layer installs a ``SoftOrthonormalConstraintRegularizer`` instead of
        leaving the kernels unregularized. Pass one explicitly to turn that
        off.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar hidden_dim: The stored branch width.
    :vartype hidden_dim: int
    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar branch_activation: The resolved branch activation, always a
        callable after ``keras.activations.get``.
    :vartype branch_activation: Callable
    :ivar gate_activation: The resolved gate activation.
    :vartype gate_activation: Callable
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the Dense layers carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer. Clones of it go
        to the five Dense layers.
    :vartype kernel_initializer: keras.initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer.
    :vartype bias_initializer: keras.initializers.Initializer
    :ivar kernel_regularizer: The regularizer actually in use. This is the
        ``SoftOrthonormalConstraintRegularizer`` default when the argument was
        None, so it is never None.
    :vartype kernel_regularizer: keras.regularizers.Regularizer
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[keras.regularizers.Regularizer]
    :ivar positive_dense: ``Dense(hidden_dim)`` on the positive path.
    :vartype positive_dense: keras.layers.Dense
    :ivar layer_norm_pos: LayerNorm on the positive path.
    :vartype layer_norm_pos: keras.layers.LayerNormalization
    :ivar positive_proj: ``Dense(hidden_dim // 2)`` on the positive path.
    :vartype positive_proj: keras.layers.Dense
    :ivar negative_dense: ``Dense(hidden_dim)`` on the negative path.
    :vartype negative_dense: keras.layers.Dense
    :ivar layer_norm_neg: LayerNorm on the negative path.
    :vartype layer_norm_neg: keras.layers.LayerNormalization
    :ivar negative_proj: ``Dense(hidden_dim // 2)`` on the negative path.
    :vartype negative_proj: keras.layers.Dense
    :ivar layer_norm_diff: LayerNorm on the difference, with ``center=False``.
    :vartype layer_norm_diff: keras.layers.LayerNormalization
    :ivar dropout: ``Dropout(dropout_rate)``. Always present.
    :vartype dropout: keras.layers.Dropout
    :ivar output_proj: ``Dense(output_dim)``, the final projection.
    :vartype output_proj: keras.layers.Dense

    :raises ValueError: If ``hidden_dim`` is not positive or is odd.
    :raises ValueError: If ``output_dim`` is not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0]``.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``.

    Output shape:
        Same rank and leading axes as the input, last axis ``output_dim``.

    Example:
        .. code-block:: python

            ffn = DifferentialFFN(hidden_dim=128, output_dim=64)
            y = ffn(keras.random.normal((2, 10, 32)))
            y.shape                 # (2, 10, 64)

    Note:
        ``hidden_dim`` must be even because each branch's second Dense maps to
        ``hidden_dim // 2``. Both branches must land on the same width for the
        subtraction to be defined.
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
        """
        Validate the configuration and create the nine sub-layers.

        Every argument is documented on the class. When ``kernel_regularizer``
        is None a ``SoftOrthonormalConstraintRegularizer`` is installed in its
        place, so the layer is never built without kernel regularization.

        :raises ValueError: If ``hidden_dim`` is not positive or is odd, if
            ``output_dim`` is not positive, or if ``dropout_rate`` is outside
            ``[0.0, 1.0]``.
        """
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

        # DECISION plan-2026-08-22T035419-a11304c8/D-200 -- clone_initializer per
        # branch. Do NOT pass the shared `self.kernel_initializer` here: the positive
        # and negative branches are the architecture, and one shared instance made
        # positive_dense.kernel == negative_dense.kernel bit-for-bit (MEASURED
        # max|delta| = 0.0), i.e. two "independent" branches that started identical.
        self.positive_dense = keras.layers.Dense(
            units=self.hidden_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
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
            # gate_activation is applied in call(), not inside the Dense.
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="positive_proj"
        )

        # Negative branch: Dense -> LayerNorm -> Activation -> Dense(gate)
        self.negative_dense = keras.layers.Dense(
            units=self.hidden_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
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
            # gate_activation is applied in call(), not inside the Dense.
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            name="negative_proj"
        )

        # Differential processing layers
        self.layer_norm_diff = keras.layers.LayerNormalization(
            # A difference is already centred at zero, so no beta is needed.
            center=False,
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
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
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
        if self.built:
            return

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
