"""
A push-pull feed-forward network, ``DifferentialFFN``, whose output is an
exactly odd function of its input: ``y(-x) = -y(x)``.

The input is split by sign into ``x_pos = ReLU(x)`` and ``x_neg = ReLU(-x)``.
Both halves are pushed through the *same* nonlinear branch map ``f``, and the
two responses are subtracted:

.. code-block:: text

    y(x) = W_out . LN_diff( f(ReLU(x)) - f(ReLU(-x)) )

Because the sign flip ``x -> -x`` swaps ``ReLU(x)`` and ``ReLU(-x)`` exactly,
and because ``f`` is shared, the difference flips sign exactly. Keeping the
post-subtraction path free of additive offsets carries that sign flip through
to the output. This is genuine push-pull: one stimulus, two channels of
opposite polarity, antagonistic at the point of combination, exactly as in
ON/OFF retinal pathways and in crustacean stretch-receptor circuits.

Why the symmetry holds (this is the whole design):

.. code-block:: text

    Let d(x) = f(ReLU(x)) - f(ReLU(-x)).

        d(-x) = f(ReLU(-x)) - f(ReLU(x)) = -d(x)          exactly, for ANY f.

    LayerNorm with center=False is odd:

        LN(-u) = g * (-u - mean(-u)) / sqrt(var(-u) + eps)
               = -g * ( u - mean( u)) / sqrt(var( u) + eps) = -LN(u)

    A bias-free Dense is linear, hence odd. Composing:

        y(-x) = W_out . LN_diff(d(-x)) = -W_out . LN_diff(d(x)) = -y(x)

Three conditions produce this, and nothing else does:

1. The two branches share weights (``tie_branches=True``). The symmetry comes
   from the *swap*, not from ``f`` being odd, so ``f`` is free to carry
   biases, LayerNorm offsets and any activation. Untying the branches
   destroys the symmetry completely; the flag exists only to ablate it.
2. ``norm_diff`` has ``center=False``. A learnable ``beta`` after the
   subtraction is an additive offset that does not flip sign, and it breaks
   oddness outright. (Note this is *not* because "a difference is already
   centred": Keras subtracts the feature-axis mean regardless of ``center``,
   and the per-sample mean of a difference is an arbitrary nonzero number.
   The flag is load-bearing here for a completely different reason.)
3. ``output_proj`` has no bias, for the same reason.

``branch_activation`` must be nonlinear. If ``f`` were affine, then
``f(x_pos) - f(x_neg) = W(x_pos - x_neg) = W x`` and the layer would collapse
to an ordinary bias-free MLP, symmetry intact but push-pull vacuous. The
LayerNorm inside the branch is already nonlinear, so the collapse is not
total, but the activation is what makes the two polarities respond
differently in shape rather than only in sign.

Cost, at ``input_dim = output_dim = d`` and ``hidden_dim = D = 4d``: about
``8 d^2`` weights, the same as a standard two-layer FFN of equal expansion,
because the branch is traversed twice but stored once.

References:
    - Shang et al., 2016. Understanding and Improving Convolutional Neural
      Networks via Concatenated Rectified Linear Units.
    - Ba et al., 2016. Layer Normalization. (https://arxiv.org/abs/1607.06450)
    - Hartline, 1938. The response of single optic nerve fibers of the
      vertebrate eye to illumination of the retina. (ON/OFF pathways)
"""

import keras
from typing import Callable, Optional, Union, Tuple, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.layers.activations.common import resolve_activation, serialize_activation

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.layers.ffn.diff_ffn")
class DifferentialFFN(keras.layers.Layer):
    """
    Push-pull feed-forward network with an exactly odd transfer function.

    The input is split by sign, both halves pass through one shared branch
    map, and the responses are subtracted. The layer satisfies
    ``y(-x) = -y(x)`` to floating point, at inference and per dropout mask
    during training. See the module docstring for the proof.

    Architecture:

    .. code-block:: text

            Input  [..., input_dim]
                     │
               ┌─────┴─────┐
               ▼           ▼
            ReLU(x)     ReLU(-x)
               │           │
               ▼           ▼
        ┌───────────────────────────┐
        │  branch_dense: Dense(D)   │   ONE set of weights,
        │  branch_norm:  LayerNorm  │   applied to both halves
        │  branch_activation        │
        └─────┬───────────────────┬─┘
              │ [..., D]          │ [..., D]
              ▼                   ▼
             push       ──►      pull
              └──── pos - neg ────┘
                     │  [..., D]
                     ▼
          ┌──────────────────────┐
          │      norm_diff       │   center=False is REQUIRED:
          │  center=False        │   a beta here breaks oddness
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │       dropout        │
          └──────────┬───────────┘
                     ▼
          ┌──────────────────────┐
          │     output_proj      │   use_bias=False is REQUIRED,
          │ Dense(output_dim)    │   same reason
          └──────────┬───────────┘
                     ▼
            Output [..., output_dim]

        D = hidden_dim, unconstrained (no parity requirement: both
        polarities traverse the same map, so they land on the same
        width by construction).

    Differences from the earlier untied, gated design:

    .. code-block:: text

        - Branch weights are tied. The old design used independent
          weights, which meant the two pathways had no relationship
          to each other and the "push-pull" name described nothing.
        - The sigmoid gate is gone. sigmoid(z) ~ 0.5 + z/4 near the
          origin, so the 0.5 offsets cancelled in the subtraction and
          norm_diff removed the 1/4; the gate was the identity at
          init and a saturating gradient sink afterwards.
        - The Dense(D) -> Dense(D/2) projections are gone. They held
          the layer's entire D^2 term and fed a bottleneck narrower
          than the residual stream at iso-parameter budget.
        - output_proj has no bias, norm_diff has no beta, and both
          are now structural requirements rather than style choices.
        - LayerNorm epsilon is explicit and defaults to 1e-6, not the
          Keras default of 1e-3. norm_diff sees a difference, which is
          the tensor most exposed to catastrophic cancellation.

        Checkpoints from the previous version will not load: the
        weight names, the layer count and the shapes all changed.

    :param hidden_dim: Width of the shared branch. Must be positive. No
        parity constraint.
    :type hidden_dim: int
    :param output_dim: Width of the output. Must be positive.
    :type output_dim: int
    :param branch_activation: Activation applied after the branch LayerNorm.
        A name ('gelu', 'relu', 'swish') or a callable. Must be nonlinear or
        the layer degenerates; see the module docstring. Defaults to 'gelu'.
    :type branch_activation: Union[str, Callable]
    :param dropout_rate: Dropout rate on the differential features, in
        ``[0.0, 1.0)``. Defaults to 0.0. The Dropout layer exists either way.
    :type dropout_rate: float
    :param use_bias: Whether the shared branch Dense carries a bias. Defaults
        to True. This does **not** affect the symmetry, because the bias is
        applied identically to both polarities and cancels in the swap. The
        output projection is unconditionally bias-free.
    :type use_bias: bool
    :param tie_branches: Whether both polarities share one branch map.
        Defaults to True, which is the only setting that yields the push-pull
        symmetry. Set to False only to ablate it; the layer then doubles its
        branch parameters and ``y(-x) = -y(x)`` no longer holds.
    :type tie_branches: bool
    :param epsilon: Variance epsilon for all LayerNormalization sub-layers.
        Defaults to 1e-6. Raise it if the differential collapses toward zero
        in low precision.
    :type epsilon: float
    :param kernel_initializer: Initializer for the kernels. Each Dense gets
        its own clone, so no mutable seed-generator state is shared between
        sub-layers. Defaults to 'glorot_uniform'.
    :type kernel_initializer: Union[str, keras.initializers.Initializer]
    :param bias_initializer: Initializer for the biases, also cloned.
        Defaults to 'zeros'.
    :type bias_initializer: Union[str, keras.initializers.Initializer]
    :param kernel_regularizer: Regularizer for the kernels. Defaults to None.
        Cloned per Dense so that stateful regularizers such as
        ``SoftOrthonormalConstraintRegularizer`` do not share state across
        layers.
    :type kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param bias_regularizer: Regularizer for the biases. Defaults to None.
        Also cloned per Dense.
    :type bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]]
    :param kwargs: Extra arguments for ``keras.layers.Layer``.
    :type kwargs: Any

    :ivar hidden_dim: The stored branch width.
    :vartype hidden_dim: int
    :ivar output_dim: The stored output width.
    :vartype output_dim: int
    :ivar branch_activation: The resolved branch activation, a callable.
    :vartype branch_activation: Callable
    :ivar dropout_rate: The stored dropout rate.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the branch Dense carries a bias.
    :vartype use_bias: bool
    :ivar tie_branches: Whether the two polarities share weights.
    :vartype tie_branches: bool
    :ivar epsilon: The LayerNormalization variance epsilon.
    :vartype epsilon: float
    :ivar branch_dense: ``Dense(hidden_dim)``, traversed by both polarities.
    :vartype branch_dense: keras.layers.Dense
    :ivar branch_norm: LayerNorm inside the branch, traversed by both.
    :vartype branch_norm: keras.layers.LayerNormalization
    :ivar branch_dense_neg: Second ``Dense(hidden_dim)``, created only when
        ``tie_branches`` is False, otherwise ``None``.
    :vartype branch_dense_neg: Optional[keras.layers.Dense]
    :ivar branch_norm_neg: Second LayerNorm, created only when
        ``tie_branches`` is False, otherwise ``None``.
    :vartype branch_norm_neg: Optional[keras.layers.LayerNormalization]
    :ivar norm_diff: LayerNorm on the difference, ``center=False``.
    :vartype norm_diff: keras.layers.LayerNormalization
    :ivar dropout: ``Dropout(dropout_rate)``. Always present.
    :vartype dropout: keras.layers.Dropout
    :ivar output_proj: ``Dense(output_dim, use_bias=False)``.
    :vartype output_proj: keras.layers.Dense

    :raises ValueError: If ``hidden_dim`` is not positive.
    :raises ValueError: If ``output_dim`` is not positive.
    :raises ValueError: If ``dropout_rate`` is outside ``[0.0, 1.0)``.
    :raises ValueError: If ``epsilon`` is not positive.

    Input shape:
        Tensor of rank >= 2, shape ``(..., input_dim)``.

    Output shape:
        Same rank and leading axes as the input, last axis ``output_dim``.

    Example:
        .. code-block:: python

            ffn = DifferentialFFN(hidden_dim=128, output_dim=64)
            y = ffn(keras.random.normal((2, 10, 32)))
            y.shape                      # (2, 10, 64)
            ffn.verify_odd_symmetry()    # (True, 3.7e-08)

    Note:
        The layer expects sign-varying input. Placed directly after a ReLU,
        a softmax or any other non-negative producer, ``x_neg`` is
        identically zero, the pull channel emits the constant ``f(0)``, and
        the layer reduces to ``W_out . LN_diff(f(x) - f(0))`` with no
        antagonism. It belongs after a residual-stream LayerNorm, not after
        an activation. Nothing enforces this at runtime.

    Note:
        Dropout is odd per realised mask, so the symmetry is exact at
        inference and exact within any single training step where both
        ``x`` and ``-x`` see the same mask. It is not exact across
        independent stochastic forward passes.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        branch_activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = "gelu",
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        tie_branches: bool = True,
        epsilon: float = 1e-6,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "glorot_uniform",
        bias_initializer: Union[str, keras.initializers.Initializer] = "zeros",
        kernel_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        bias_regularizer: Optional[Union[str, keras.regularizers.Regularizer]] = None,
        **kwargs: Any
    ) -> None:
        """
        Validate the configuration and create the sub-layers.

        Creates four sub-layers when ``tie_branches`` is True and six when it
        is False. Every argument is documented on the class.

        :raises ValueError: If ``hidden_dim`` or ``output_dim`` is not
            positive, if ``dropout_rate`` is outside ``[0.0, 1.0)``, or if
            ``epsilon`` is not positive.
        """
        super().__init__(**kwargs)

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError(f"dropout_rate must be in [0.0, 1.0), got {dropout_rate}")
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.branch_activation = resolve_activation(branch_activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.tie_branches = tie_branches
        self.epsilon = epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)

        # ---- the shared push-pull branch -----------------------------
        # One Dense and one LayerNorm, traversed twice in call(). Tying
        # these is what makes the layer odd: the sign flip x -> -x swaps
        # the two ReLU halves, and a shared map turns that swap into an
        # exact sign flip of the difference. Independent weights would
        # leave the two polarities unrelated.
        #
        # Initializers and regularizers are cloned per sub-layer so that
        # no mutable seed-generator or regularizer state is shared. Note
        # this does NOT decorrelate an untied pair under a seeded
        # initializer: clone_initializer round-trips get_config, which
        # preserves the seed, so a seeded clone reproduces its source
        # bit for bit. Decorrelation is not needed in either case, since
        # the branches consume disjoint halves of the input.
        self.branch_dense = keras.layers.Dense(
            units=self.hidden_dim,
            activation=None,
            use_bias=self.use_bias,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            kernel_regularizer=self._clone_regularizer(self.kernel_regularizer),
            bias_regularizer=self._clone_regularizer(self.bias_regularizer),
            name="branch_dense"
        )
        self.branch_norm = keras.layers.LayerNormalization(
            epsilon=self.epsilon,
            center=True,
            scale=True,
            name="branch_norm"
        )

        # ---- optional untied pull branch (ablation only) --------------
        self.branch_dense_neg: Optional[keras.layers.Dense] = None
        self.branch_norm_neg: Optional[keras.layers.LayerNormalization] = None
        if not self.tie_branches:
            self.branch_dense_neg = keras.layers.Dense(
                units=self.hidden_dim,
                activation=None,
                use_bias=self.use_bias,
                kernel_initializer=clone_initializer(self.kernel_initializer),
                bias_initializer=clone_initializer(self.bias_initializer),
                kernel_regularizer=self._clone_regularizer(self.kernel_regularizer),
                bias_regularizer=self._clone_regularizer(self.bias_regularizer),
                name="branch_dense_neg"
            )
            self.branch_norm_neg = keras.layers.LayerNormalization(
                epsilon=self.epsilon,
                center=True,
                scale=True,
                name="branch_norm_neg"
            )

        # ---- post-subtraction path -----------------------------------
        # center=False is a hard requirement, not a stylistic one. Keras
        # subtracts the feature-axis mean whether or not center is set;
        # the flag controls only the learnable beta offset. A beta here
        # is an additive constant that survives the sign flip unchanged
        # and would destroy y(-x) = -y(x).
        self.norm_diff = keras.layers.LayerNormalization(
            epsilon=self.epsilon,
            center=False,
            scale=True,
            name="norm_diff"
        )
        self.dropout = keras.layers.Dropout(
            rate=self.dropout_rate,
            name="dropout"
        )
        # use_bias=False for the same reason as center=False above.
        self.output_proj = keras.layers.Dense(
            units=self.output_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            kernel_regularizer=self._clone_regularizer(self.kernel_regularizer),
            name="output_proj"
        )

    @staticmethod
    def _clone_regularizer(
        regularizer: Optional[keras.regularizers.Regularizer]
    ) -> Optional[keras.regularizers.Regularizer]:
        """
        Return an independent copy of a regularizer, or ``None``.

        Sharing one regularizer instance across sub-layers is safe for the
        stateless built-ins but not for stateful ones such as
        ``SoftOrthonormalConstraintRegularizer``. Cloning costs nothing and
        removes the question.

        :param regularizer: The regularizer to copy, or ``None``.
        :type regularizer: Optional[keras.regularizers.Regularizer]
        :return: A fresh instance with the same configuration, or ``None``.
        :rtype: Optional[keras.regularizers.Regularizer]
        """
        if regularizer is None:
            return None
        return regularizer.__class__.from_config(regularizer.get_config())

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all its sub-layers.

        The shared branch is built once against ``input_shape``; both
        polarities have the same shape, so a single build covers both
        traversals.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the last axis of ``input_shape`` is undefined.
        """
        if self.built:
            return

        if input_shape[-1] is None:
            raise ValueError(
                "The last axis of the input shape must be defined, "
                f"got {input_shape}"
            )

        self.branch_dense.build(input_shape)
        branch_shape = self.branch_dense.compute_output_shape(input_shape)
        self.branch_norm.build(branch_shape)

        if not self.tie_branches:
            self.branch_dense_neg.build(input_shape)
            self.branch_norm_neg.build(branch_shape)

        self.norm_diff.build(branch_shape)
        self.dropout.build(branch_shape)
        self.output_proj.build(branch_shape)

        super().build(input_shape)

    def _branch(
        self,
        x: keras.KerasTensor,
        negative: bool = False
    ) -> keras.KerasTensor:
        """
        Apply the branch map ``f`` to one non-negative polarity.

        With ``tie_branches`` True, ``negative`` is ignored and both calls go
        through the same weights, which is what produces the symmetry.

        :param x: Non-negative tensor, shape ``(..., input_dim)``.
        :type x: keras.KerasTensor
        :param negative: Whether to route through the untied pull branch.
            Only meaningful when ``tie_branches`` is False.
        :type negative: bool
        :return: Branch response, shape ``(..., hidden_dim)``.
        :rtype: keras.KerasTensor
        """
        if negative and not self.tie_branches:
            return self.branch_activation(self.branch_norm_neg(self.branch_dense_neg(x)))
        return self.branch_activation(self.branch_norm(self.branch_dense(x)))

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass: split by sign, push both polarities through the shared
        branch, subtract, normalize, project.

        :param inputs: Input tensor with shape ``(..., input_dim)``.
        :type inputs: keras.KerasTensor
        :param training: Whether to apply dropout.
        :type training: Optional[bool]
        :return: Output tensor with shape ``(..., output_dim)``.
        :rtype: keras.KerasTensor
        """
        # The sign split is exact and exhaustive: x = x_pos - x_neg, and
        # x -> -x swaps the two. That swap is the source of the symmetry.
        x_pos = keras.ops.relu(inputs)
        x_neg = keras.ops.relu(-inputs)

        push = self._branch(x_pos)
        pull = self._branch(x_neg, negative=True)

        # Antagonistic combination. This is not a weaker form of concat:
        # the subtraction is the operation that carries the sign flip, and
        # replacing it with concat plus a learned projection would discard
        # the symmetry entirely.
        differential = push - pull

        # No training kwarg: LayerNormalization has no training-dependent
        # behaviour, and passing one implies a dependence that is not there.
        normed = self.norm_diff(differential)
        dropped = self.dropout(normed, training=training)

        return self.output_proj(dropped)

    def verify_odd_symmetry(
        self,
        input_shape: Optional[Tuple[int, ...]] = None,
        tolerance: float = 1e-5,
        seed: int = 0
    ) -> Tuple[bool, float]:
        """
        Check ``y(-x) == -y(x)`` numerically on random input.

        A debug helper for confirming that the push-pull property survives
        subclassing, weight surgery or a configuration change. Runs in
        inference mode, so dropout is inactive.

        :param input_shape: Shape to probe with. Defaults to a small batch
            matching the built input width; required if the layer is unbuilt.
        :type input_shape: Optional[Tuple[int, ...]]
        :param tolerance: Maximum acceptable absolute deviation.
        :type tolerance: float
        :param seed: Seed for the probe tensor.
        :type seed: int
        :return: ``(passed, max_absolute_deviation)``.
        :rtype: Tuple[bool, float]
        :raises ValueError: If the layer is unbuilt and no shape is given.
        """
        if input_shape is None:
            if not self.built:
                raise ValueError(
                    "Layer is not built; pass input_shape to verify_odd_symmetry."
                )
            input_shape = (4, self.branch_dense.kernel.shape[0])

        x = keras.random.normal(input_shape, seed=seed)
        y_pos = self(x, training=False)
        y_neg = self(-x, training=False)

        deviation = float(keras.ops.max(keras.ops.abs(y_neg + y_pos)))
        return deviation <= tolerance, deviation

    def compute_output_shape(
        self,
        input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape of the layer.

        :param input_shape: Shape tuple of the input.
        :type input_shape: Tuple[Optional[int], ...]
        :return: Output shape with the last axis replaced by ``output_dim``.
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
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "branch_activation": serialize_activation(self.branch_activation),
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
            "tie_branches": self.tie_branches,
            "epsilon": self.epsilon,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------