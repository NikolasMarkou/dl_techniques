"""
The MLP-Mixer block.

This is the mixing block of MLP-Mixer (Tolstikhin et al. 2021), an
attention-free architecture that processes a sequence of tokens or image
patches with nothing but multi-layer perceptrons. It takes a rank-3 tensor
``(B, S, C)``, where ``S`` is the token count and ``C`` the channel count,
and returns the same shape.

Two mixing operations run in sequence, each inside its own pre-LayerNorm
residual sub-block:

1.  **Token-mixing** mixes across the token axis ``S``, one channel at a
    time. The input is normalized, transposed to ``(B, C, S)`` so that an
    MLP acting on the last axis mixes tokens, transposed back, and added to
    the residual stream.

2.  **Channel-mixing** mixes across the channel axis ``C``, one token at a
    time. The input is normalized, an MLP acts on the last axis directly,
    and the result is added to the residual stream.

Each mixing MLP is a two-layer expand-then-contract perceptron with one
non-linearity (GELU by default): ``Dense(hidden) -> activation -> Dropout ->
Dense(restore)``. ``restore`` is ``S`` for the token MLP and ``C`` for the
channel MLP. Doing both in sequence gives every position a path to every
other, which is the job attention would otherwise do.

**Mathematics:**
Let ``X`` be the input of shape ``(B, S, C)``, ``LN`` a per-channel layer
normalization, ``W_*`` the MLP weight matrices and ``sigma`` the activation:

1.  Token-mixing, along the ``S`` axis, per channel:
    ``U = X + (sigma(LN(X)^T @ W_1) @ W_2)^T``
    The transpose moves the token axis last so the Dense layers mix tokens.
    ``W_1`` is ``(S, tokens_mlp_dim)`` and ``W_2`` is ``(tokens_mlp_dim, S)``.

2.  Channel-mixing, along the ``C`` axis, per token:
    ``Y = U + sigma(LN(U) @ W_3) @ W_4``
    ``W_3`` is ``(C, channels_mlp_dim)`` and ``W_4`` is
    ``(channels_mlp_dim, C)``.

The token MLP's output width is ``S`` and the channel MLP's is ``C``. Neither
is known until an input shape arrives, so the two Dense layers that need them
are created in ``build()``, the same deferred pattern ``SwinMLP`` uses. Every
other sublayer is fully determined by the constructor arguments and is
created in ``__init__``.

References:
-   Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X.,
    Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J.,
    Lucic, M., & Dosovitskiy, A. (2021). MLP-Mixer: An all-MLP Architecture
    for Vision. arXiv preprint arXiv:2105.01601.

"""

import keras
from typing import Callable, Optional, Union, Any, Dict, Tuple
from keras import layers, initializers, regularizers, activations

from dl_techniques.initializers.clone import clone_initializer
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.layers.ffn.mlp_mixer_block")
class MixerBlock(keras.layers.Layer):
    """
    The MLP-Mixer block: token-mixing, then channel-mixing.

    Takes a rank-3 tensor ``(B, S, C)`` and returns the same shape. ``S`` is
    the number of tokens or patches, ``C`` the number of channels. There is
    no attention here. Two MLPs do all the mixing, one along each axis, and
    each sits inside a pre-norm residual sub-block.

    Token-mixing mixes across ``S``, one channel at a time. Channel-mixing
    mixes across ``C``, one token at a time. Applying both in sequence gives
    every position a path to every other, which is what attention would have
    provided.

    **Architecture Overview:**

    .. code-block:: text

        Input  [B, S, C]
                     │
                     ├─────────────────┐
                     ▼                 │
        ┌──────────────────────────┐   │
        │ token-mixing sub-block   │   │
        │ LN -> T -> MLP -> T      │   │
        └────────────┬─────────────┘   │
                     ▼                 │
                    (+) ◄──────────────┘
                     │
                     ├─────────────────┐
                     ▼                 │
        ┌──────────────────────────┐   │
        │ channel-mixing sub-block │   │
        │ LN -> MLP                │   │
        └────────────┬─────────────┘   │
                     ▼                 │
                    (+) ◄──────────────┘
                     │
                     ▼
        Output [B, S, C]

        Every edge on the main path carries [B, S, C]. Both
        sub-blocks are pre-norm: the LayerNormalization sits
        inside the branch, and the residual carries the
        unnormalized tensor. `T` is a transpose.

    **Token-mixing sub-block:**

    .. code-block:: text

        x  [B, S, C]
             │
             ▼
        token_norm     LayerNormalization
             │  [B, S, C]
             ▼
        transpose (0, 2, 1)
             │  [B, C, S]
             ▼
        token_mlp_hidden   Dense(tokens_mlp_dim)
             │  [B, C, tokens_mlp_dim]
             ▼
        activation
             │
             ▼
        token_dropout      Dropout
             │
             ▼
        token_mlp_out      Dense(S)
             │  [B, C, S]
             ▼
        transpose (0, 2, 1)
             │  [B, S, C]
             ▼
        y, added to x

        The two transposes are what people get wrong. Dense
        always acts on the LAST axis, so S is moved there and
        moved back. The pair must stay balanced: drop either
        one and the block stops being shape-preserving.

        Dense(S) means units=S. The token count is a WEIGHT
        SHAPE, which is why S has to be known at build time.

    **Channel-mixing sub-block:**

    .. code-block:: text

        x  [B, S, C]
             │
             ▼
        channel_norm     LayerNormalization
             │  [B, S, C]
             ▼
        channel_mlp_hidden  Dense(channels_mlp_dim)
             │  [B, S, channels_mlp_dim]
             ▼
        activation
             │
             ▼
        channel_dropout     Dropout
             │
             ▼
        channel_mlp_out     Dense(C)
             │  [B, S, C]
             ▼
        z, added to x

        No transpose. C is already the last axis. Dense(C)
        means units=C, so C must be known at build time too.

    :param tokens_mlp_dim: Hidden width of the token-mixing MLP, the MLP that
        runs across the token axis ``S``. Must be a positive int.
    :type tokens_mlp_dim: int
    :param channels_mlp_dim: Hidden width of the channel-mixing MLP, the MLP
        that runs across the channel axis ``C``. Must be a positive int.
    :type channels_mlp_dim: int
    :param activation: Activation used inside both mixing MLPs. A Keras name
        ('gelu', 'relu', ...) or a callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Rate for both dropouts, in ``[0, 1]``. The dropouts
        sit after the activation in each MLP. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the four Dense layers carry a bias. Defaults to
        True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for all four Dense kernels. Each
        of the four receives its own clone of it. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for all four Dense biases, cloned
        per layer in the same way. Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Regularizer for all Dense kernels. Defaults to
        None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Regularizer for all Dense biases. Defaults to
        None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Extra arguments for ``keras.layers.Layer`` (``name``,
        ``dtype``, and so on).
    :type kwargs: Any

    :ivar tokens_mlp_dim: The stored token-MLP hidden width.
    :vartype tokens_mlp_dim: int
    :ivar channels_mlp_dim: The stored channel-MLP hidden width.
    :vartype channels_mlp_dim: int
    :ivar activation: The resolved activation, called directly in ``call()``.
    :vartype activation: Callable
    :ivar dropout_rate: The stored dropout rate, shared by both dropouts.
    :vartype dropout_rate: float
    :ivar use_bias: Whether the Dense layers carry a bias.
    :vartype use_bias: bool
    :ivar kernel_initializer: The resolved kernel initializer. It is the
        source the four per-layer clones are rebuilt from, and is not handed
        to any Dense layer itself.
    :vartype kernel_initializer: initializers.Initializer
    :ivar bias_initializer: The resolved bias initializer, cloned per layer
        in the same way.
    :vartype bias_initializer: initializers.Initializer
    :ivar kernel_regularizer: The resolved kernel regularizer, or ``None``.
    :vartype kernel_regularizer: Optional[regularizers.Regularizer]
    :ivar bias_regularizer: The resolved bias regularizer, or ``None``.
    :vartype bias_regularizer: Optional[regularizers.Regularizer]
    :ivar _dense_kwargs: The keyword dict handed to every Dense layer. Kept
        so ``build()`` can construct the two back-projections the same way
        ``__init__`` constructed the two hidden projections.
    :vartype _dense_kwargs: Dict[str, Any]
    :ivar token_norm: Pre-norm of the token-mixing sub-block.
    :vartype token_norm: keras.layers.LayerNormalization
    :ivar channel_norm: Pre-norm of the channel-mixing sub-block.
    :vartype channel_norm: keras.layers.LayerNormalization
    :ivar token_mlp_hidden: ``Dense(tokens_mlp_dim)``.
    :vartype token_mlp_hidden: keras.layers.Dense
    :ivar channel_mlp_hidden: ``Dense(channels_mlp_dim)``.
    :vartype channel_mlp_hidden: keras.layers.Dense
    :ivar token_dropout: Dropout inside the token MLP.
    :vartype token_dropout: keras.layers.Dropout
    :ivar channel_dropout: Dropout inside the channel MLP.
    :vartype channel_dropout: keras.layers.Dropout
    :ivar token_mlp_out: ``Dense(S)``, the token back-projection. ``None``
        until ``build()`` runs.
    :vartype token_mlp_out: Optional[keras.layers.Dense]
    :ivar channel_mlp_out: ``Dense(C)``, the channel back-projection.
        ``None`` until ``build()`` runs.
    :vartype channel_mlp_out: Optional[keras.layers.Dense]

    :raises ValueError: If ``tokens_mlp_dim`` or ``channels_mlp_dim`` is not
        a positive int.
    :raises ValueError: If ``dropout_rate`` is outside ``[0, 1]``.
    :raises ValueError: At build time, if the input is not rank 3, or if
        ``S`` or ``C`` is ``None``.

    Input shape:
        Rank-3 tensor ``(B, S, C)``. Both ``S`` and ``C`` must be statically
        known. A rank other than 3 is rejected in ``build()``.

    Output shape:
        ``(B, S, C)``, identical to the input.

    Example:
        .. code-block:: python

            block = MixerBlock(
                tokens_mlp_dim=256, channels_mlp_dim=2048)
            y = block(keras.random.normal((2, 196, 512)))
            y.shape                 # (2, 196, 512)

    Note:
        This is the only layer in ``ffn/`` that pins the input rank to 3
        with an explicit check. ``TverskyProjectionLayer`` pins its own to 2;
        ``SwinMLP``, ``KANLinear``, ``CountingFFN`` and ``LogicFFN`` only
        require rank 2 or more. ``GatedMLP`` is rank 4 only, enforced by
        Conv2D rather than by a check: a rank-3 input raises from the
        convolution, not from a guard. The reason the rank is pinned here is
        the two back-projections: ``token_mlp_out`` has ``units=S`` and
        ``channel_mlp_out`` has ``units=C``, so both axes are weight shapes
        and both must be statically known.
    """

    def __init__(
        self,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        activation: Union[str, Callable[[keras.KerasTensor], keras.KerasTensor]] = 'gelu',
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[regularizers.Regularizer] = None,
        bias_regularizer: Optional[regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        """Validate the configuration and create the sub-layers.

        Every argument is documented on the class. Validation runs before any
        attribute is stored, so a rejected configuration leaves no half-built
        layer behind.

        The two back-projections are NOT created here. Their widths are ``S``
        and ``C``, which only ``build()`` can read off the input shape. The
        shared Dense keyword dict is stashed on ``_dense_kwargs`` so
        ``build()`` can construct them the same way. It carries the
        regularizers and ``use_bias`` only; the initializers are cloned at
        each Dense call site instead.

        :param tokens_mlp_dim: Hidden width of the token-mixing MLP. Must be
            a positive int.
        :type tokens_mlp_dim: int
        :param channels_mlp_dim: Hidden width of the channel-mixing MLP. Must
            be a positive int.
        :type channels_mlp_dim: int
        :param activation: Activation used inside both mixing MLPs.
        :type activation: Union[str, Callable]
        :param dropout_rate: Rate for both dropouts. Must be in ``[0, 1]``.
        :type dropout_rate: float
        :param use_bias: Whether the four Dense layers carry a bias.
        :type use_bias: bool
        :param kernel_initializer: Initializer for all four kernels, cloned
            once per kernel.
        :type kernel_initializer: Union[str, initializers.Initializer]
        :param bias_initializer: Initializer for all four biases, cloned once
            per bias.
        :type bias_initializer: Union[str, initializers.Initializer]
        :param kernel_regularizer: Regularizer for all kernels, or ``None``.
        :type kernel_regularizer: Optional[regularizers.Regularizer]
        :param bias_regularizer: Regularizer for all biases, or ``None``.
        :type bias_regularizer: Optional[regularizers.Regularizer]
        :param kwargs: Extra arguments for ``keras.layers.Layer``.
        :type kwargs: Any

        :raises ValueError: If ``tokens_mlp_dim`` or ``channels_mlp_dim`` is
            not a positive int, or if ``dropout_rate`` is outside ``[0, 1]``.
        """
        super().__init__(**kwargs)

        # Comprehensive input validation with informative error messages
        if not isinstance(tokens_mlp_dim, int) or tokens_mlp_dim <= 0:
            raise ValueError(f"tokens_mlp_dim must be a positive integer, got {tokens_mlp_dim}")
        if not isinstance(channels_mlp_dim, int) or channels_mlp_dim <= 0:
            raise ValueError(f"channels_mlp_dim must be a positive integer, got {channels_mlp_dim}")
        if not isinstance(dropout_rate, (int, float)) or not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")

        # Store ALL configuration parameters for serialization
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.activation = activations.get(activation)
        self.dropout_rate = float(dropout_rate)
        self.use_bias = bool(use_bias)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # The stashed dict carries NO initializer. Each of the four Dense
        # layers takes its own clone instead; the rule and the mechanism are
        # written out at glu_ffn.py, decisions.md D-008. The token and
        # channel MLPs collide pairwise at tokens_mlp_dim == channels_mlp_dim
        # (hidden kernels and biases) and at S == C (the two back-projections)
        # -- MEASURED max|delta| = 0.0 at 8/8 over an (8, 8) input, which
        # made the two mixing directions the same function at init.
        dense_kwargs = {
            "use_bias": self.use_bias,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        }
        self._dense_kwargs = dense_kwargs

        # ---- Sublayers whose configuration is fully known from the ctor ----
        # Pre-LN for each residual sub-block.
        self.token_norm = layers.LayerNormalization(name="token_norm")
        self.channel_norm = layers.LayerNormalization(name="channel_norm")

        # Token-mixing MLP hidden projection (units = tokens_mlp_dim, known now).
        self.token_mlp_hidden = layers.Dense(
            self.tokens_mlp_dim,
            activation=None,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="token_mlp_hidden",
            **dense_kwargs
        )
        # Channel-mixing MLP hidden projection (units = channels_mlp_dim, known now).
        self.channel_mlp_hidden = layers.Dense(
            self.channels_mlp_dim,
            activation=None,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="channel_mlp_hidden",
            **dense_kwargs
        )

        # Dropout (rate known now). One per mixing MLP for clean serialization.
        self.token_dropout = layers.Dropout(rate=self.dropout_rate, name="token_dropout")
        self.channel_dropout = layers.Dropout(rate=self.dropout_rate, name="channel_dropout")

        # ---- Sublayers whose units depend on S or C: created in build() ----
        # units = S (tokens) and units = C (channels). Neither is known yet.
        self.token_mlp_out = None
        self.channel_mlp_out = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Read ``S`` and ``C`` off the input shape, then build everything.

        The two back-projections cannot exist before this point:
        ``token_mlp_out`` has ``units=S`` and ``channel_mlp_out`` has
        ``units=C``, and neither number is known until an input shape
        arrives. They are created here, then every sub-layer is built in the
        order ``call()`` uses them, so all weights exist before a save or a
        restore.

        The ``self.built`` guard on the first line keeps this safe: without
        it a second ``build()`` would replace both Dense layers and drop
        their weights.

        :param input_shape: Shape tuple of the input tensor. Must be rank 3.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank 3, or if ``S`` or ``C``
            is ``None``.
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(
                f"MixerBlock requires rank-3 input (B,S,C), got shape {input_shape}"
            )

        # S = tokens, C = channels.
        seq_len = input_shape[1]
        channels = input_shape[2]
        if seq_len is None:
            raise ValueError(
                f"The token dimension S (input_shape[1]) must be statically "
                f"defined, got shape {input_shape}"
            )
        if channels is None:
            raise ValueError(
                f"The channel dimension C (input_shape[2]) must be statically "
                f"defined, got shape {input_shape}"
            )

        # Create the dimension-dependent back-projections now that S, C are known.
        self.token_mlp_out = layers.Dense(
            seq_len,
            activation=None,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="token_mlp_out",
            **self._dense_kwargs
        )
        self.channel_mlp_out = layers.Dense(
            channels,
            activation=None,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            name="channel_mlp_out",
            **self._dense_kwargs
        )

        # ---- Token-mixing sub-block (operates on transposed (B, C, S)) ----
        self.token_norm.build(input_shape)
        # After LN + transpose, the token MLP sees (B, C, S).
        transposed_shape = (input_shape[0], channels, seq_len)
        self.token_mlp_hidden.build(transposed_shape)
        token_hidden_shape = self.token_mlp_hidden.compute_output_shape(transposed_shape)
        self.token_dropout.build(token_hidden_shape)
        self.token_mlp_out.build(token_hidden_shape)

        # ---- Channel-mixing sub-block (operates on (B, S, C)) ----
        self.channel_norm.build(input_shape)
        self.channel_mlp_hidden.build(input_shape)
        channel_hidden_shape = self.channel_mlp_hidden.compute_output_shape(input_shape)
        self.channel_dropout.build(channel_hidden_shape)
        self.channel_mlp_out.build(channel_hidden_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Run the token-mixing residual, then the channel-mixing residual.

        Both sub-blocks are pre-norm: the LayerNormalization sits inside the
        branch, and the residual carries the unnormalized tensor forward.

        :param inputs: Input tensor of shape ``(B, S, C)``.
        :type inputs: keras.KerasTensor
        :param training: Whether to run in training mode. Only affects the
            two dropouts.
        :type training: Optional[bool]
        :return: Output tensor of shape ``(B, S, C)``.
        :rtype: keras.KerasTensor
        """
        # ---- Token-mixing sub-block ----
        # Shapes down this branch: (B, S, C) -> (B, C, S) ->
        # (B, C, tokens_mlp_dim) -> (B, C, S) -> (B, S, C).
        y = self.token_norm(inputs)
        y = keras.ops.transpose(y, axes=(0, 2, 1))
        y = self.token_mlp_hidden(y)
        y = self.activation(y)
        y = self.token_dropout(y, training=training)
        y = self.token_mlp_out(y)
        y = keras.ops.transpose(y, axes=(0, 2, 1))
        # Residual add.
        x = inputs + y

        # ---- Channel-mixing sub-block ----
        # Shapes down this branch: (B, S, C) ->
        # (B, S, channels_mlp_dim) -> (B, S, C). No transpose here.
        z = self.channel_norm(x)
        z = self.channel_mlp_hidden(z)
        z = self.activation(z)
        z = self.channel_dropout(z, training=training)
        z = self.channel_mlp_out(z)
        # Residual add.
        output = x + z

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape, which equals the input shape.

        Both sub-blocks are residual and both project back to the width they
        started from, so the block is shape-preserving.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The input shape, unchanged.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Return the config needed to rebuild this layer.

        Holds every ``__init__`` argument. The two width-dependent Dense
        layers are not in here: their widths come from the input shape, so a
        reloaded block re-derives them in ``build()``.

        :return: The complete layer configuration.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            'tokens_mlp_dim': self.tokens_mlp_dim,
            'channels_mlp_dim': self.channels_mlp_dim,
            'activation': activations.serialize(self.activation),
            'dropout_rate': self.dropout_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        })
        return config

# ---------------------------------------------------------------------
