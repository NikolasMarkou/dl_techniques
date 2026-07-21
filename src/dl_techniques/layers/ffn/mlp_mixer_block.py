"""
A canonical MLP-Mixer block.

This layer implements the core mixing block of the MLP-Mixer architecture
(Tolstikhin et al. 2021), an attention-free model that processes a sequence of
tokens (or image patches) using only multi-layer perceptrons. It operates on a
rank-3 tensor ``(B, S, C)`` where ``S`` is the number of tokens/patches and
``C`` is the channel dimension, and returns a tensor of the same shape.

The block interleaves two complementary mixing operations, each wrapped in a
pre-LayerNorm residual sub-block:

1.  **Token-mixing** mixes information *across* the token axis ``S``,
    independently for every channel. The input is layer-normalized and
    transposed to ``(B, C, S)`` so that an MLP applied to the last axis mixes
    tokens. The result is transposed back to ``(B, S, C)`` and added to the
    residual stream.

2.  **Channel-mixing** mixes information *across* the channel axis ``C``,
    independently for every token. The input is layer-normalized and an MLP is
    applied directly to the last axis ``C``, then added to the residual stream.

Each mixing MLP is a two-layer expand-then-contract perceptron with a single
non-linearity (GELU by default): ``Dense(hidden) -> activation -> Dropout ->
Dense(restore)``, where ``restore`` equals ``S`` for the token MLP and ``C``
for the channel MLP. This symmetric token/channel decomposition is the central
idea of MLP-Mixer: it replaces self-attention with two cheap dense mixers while
keeping a fixed-size receptive field over both axes.

Architectural Overview:

.. code-block:: text

    Input (B, S, C)
        │
        ├───────────────────────────────┐ (residual)
        ▼                               │
    LayerNorm (token_norm)              │
        ▼                               │
    transpose -> (B, C, S)              │
        ▼                               │
    Dense(tokens_mlp_dim) -> act        │
        ▼                               │
    Dropout -> Dense(S)                 │
        ▼                               │
    transpose -> (B, S, C)              │
        ▼                               │
       (+) ◄────────────────────────────┘
        │
        ├───────────────────────────────┐ (residual)
        ▼                               │
    LayerNorm (channel_norm)            │
        ▼                               │
    Dense(channels_mlp_dim) -> act      │
        ▼                               │
    Dropout -> Dense(C)                 │
        ▼                               │
       (+) ◄────────────────────────────┘
        │
        ▼
    Output (B, S, C)

Foundational Mathematics:
Let ``X`` be the input of shape ``(B, S, C)``. With ``LN`` a per-channel layer
normalization, ``W_*`` the MLP weight matrices and ``sigma`` the activation,
the block computes:

1.  Token-mixing (applied along the ``S`` axis, per channel):
    ``U = X + (sigma(LN(X)^T @ W_1) @ W_2)^T``
    where the transpose moves the token axis to the last position so the dense
    layers mix tokens. ``W_1`` has shape ``(S, tokens_mlp_dim)`` and ``W_2`` has
    shape ``(tokens_mlp_dim, S)``.

2.  Channel-mixing (applied along the ``C`` axis, per token):
    ``Y = U + sigma(LN(U) @ W_3) @ W_4``
    where ``W_3`` has shape ``(C, channels_mlp_dim)`` and ``W_4`` has shape
    ``(channels_mlp_dim, C)``.

Because the token MLP's output width is ``S`` and the channel MLP's output
width is ``C``, both of which are only known once the input shape is seen, the
``Dense`` layers whose ``units`` depend on ``S`` or ``C`` are created in
``build()`` (the deferred-creation pattern used by ``SwinMLP``), while every
sublayer whose configuration is fully determined by the constructor arguments
is created in ``__init__``.

References:
-   Tolstikhin, I., Houlsby, N., Kolesnikov, A., Beyer, L., Zhai, X.,
    Unterthiner, T., Yung, J., Steiner, A., Keysers, D., Uszkoreit, J.,
    Lucic, M., & Dosovitskiy, A. (2021). MLP-Mixer: An all-MLP Architecture
    for Vision. arXiv preprint arXiv:2105.01601.

"""

import keras
from typing import Callable, Optional, Union, Any, Dict, Tuple
from keras import layers, initializers, regularizers, activations

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MixerBlock(keras.layers.Layer):
    """
    Canonical MLP-Mixer block (token-mixing then channel-mixing).

    Operates on rank-3 inputs ``(B, S, C)`` (S = tokens/patches, C = channels)
    and returns a tensor of the same shape. It applies a pre-LayerNorm residual
    token-mixing sub-block (MLP over the token axis via a transpose) followed by
    a pre-LayerNorm residual channel-mixing sub-block (MLP over the channel
    axis), as introduced by Tolstikhin et al. (2021).

    **Architecture Overview:**

    .. code-block:: text

        Input (B, S, C)
          → [ +  LayerNorm → transpose(B,C,S) → Dense(tokens_mlp_dim) → act
                 → Dropout → Dense(S) → transpose(B,S,C) ]   (token-mixing residual)
          → [ +  LayerNorm → Dense(channels_mlp_dim) → act
                 → Dropout → Dense(C) ]                       (channel-mixing residual)
        Output (B, S, C)

    :param tokens_mlp_dim: Integer, hidden width of the token-mixing MLP (the
        MLP applied across the token axis ``S``). Must be positive.
    :type tokens_mlp_dim: int
    :param channels_mlp_dim: Integer, hidden width of the channel-mixing MLP
        (the MLP applied across the channel axis ``C``). Must be positive.
    :type channels_mlp_dim: int
    :param activation: Activation applied inside both mixing MLPs. Can be a
        string name ('gelu', 'relu', ...) or a callable. Defaults to 'gelu'.
    :type activation: Union[str, Callable]
    :param dropout_rate: Float in [0, 1], dropout applied after the activation
        in both mixing MLPs. Only active during training. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether the Dense layers include bias terms. Defaults to True.
    :type use_bias: bool
    :param kernel_initializer: Initializer for all Dense kernels. Defaults to
        'glorot_uniform'.
    :type kernel_initializer: Union[str, initializers.Initializer]
    :param bias_initializer: Initializer for all Dense biases. Defaults to 'zeros'.
    :type bias_initializer: Union[str, initializers.Initializer]
    :param kernel_regularizer: Optional regularizer for all Dense kernels.
        Defaults to None.
    :type kernel_regularizer: Optional[regularizers.Regularizer]
    :param bias_regularizer: Optional regularizer for all Dense biases.
        Defaults to None.
    :type bias_regularizer: Optional[regularizers.Regularizer]
    :param kwargs: Additional keyword arguments for the Layer base class.

    :raises ValueError: If tokens_mlp_dim or channels_mlp_dim are not positive integers.
    :raises ValueError: If dropout_rate is not between 0 and 1.
    :raises ValueError: At build time, if the input is not rank-3 or if the token
        (``S``) or channel (``C``) dimension is not statically known.

    Note:
        This is the only ffn/ layer that constrains the input rank. Both the
        token (``S``) and channel (``C``) dimensions must be statically defined
        on the input, because the token-mixing back-projection has ``units=S``
        and the channel-mixing back-projection has ``units=C``.
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
        """Initialize the MixerBlock with comprehensive parameter validation."""
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

        dense_kwargs = {
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
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
            name="token_mlp_hidden",
            **dense_kwargs
        )
        # Channel-mixing MLP hidden projection (units = channels_mlp_dim, known now).
        self.channel_mlp_hidden = layers.Dense(
            self.channels_mlp_dim,
            activation=None,
            name="channel_mlp_hidden",
            **dense_kwargs
        )

        # Dropout (rate known now). One per mixing MLP for clean serialization.
        self.token_dropout = layers.Dropout(rate=self.dropout_rate, name="token_dropout")
        self.channel_dropout = layers.Dropout(rate=self.dropout_rate, name="channel_dropout")

        # ---- Sublayers whose units depend on S or C: created in build() ----
        self.token_mlp_out = None    # units = S (number of tokens)
        self.channel_mlp_out = None  # units = C (number of channels)

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer and all sub-layers in computational order.

        Validates the rank-3 contract, infers ``S`` and ``C`` from the input
        shape, creates the dimension-dependent back-projection Dense layers, and
        explicitly builds every sublayer so all weights exist before any
        serialization restore. ``super().build()`` is called last.

        :param input_shape: Shape tuple of the input tensor, must be rank-3.
        :type input_shape: Tuple[Optional[int], ...]
        :raises ValueError: If the input is not rank-3, or if S or C is undefined.
        """
        if self.built:
            return

        if len(input_shape) != 3:
            raise ValueError(
                f"MixerBlock requires rank-3 input (B,S,C), got shape {input_shape}"
            )

        seq_len = input_shape[1]   # S (tokens)
        channels = input_shape[2]  # C (channels)
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
            name="token_mlp_out",
            **self._dense_kwargs
        )
        self.channel_mlp_out = layers.Dense(
            channels,
            activation=None,
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
        Forward pass: token-mixing residual then channel-mixing residual.

        :param inputs: Input tensor of shape (B, S, C).
        :type inputs: keras.KerasTensor
        :param training: Whether the layer is in training mode (affects dropout).
        :type training: Optional[bool]
        :return: Output tensor of shape (B, S, C).
        :rtype: keras.KerasTensor
        """
        # ---- Token-mixing sub-block ----
        y = self.token_norm(inputs)
        y = keras.ops.transpose(y, axes=(0, 2, 1))   # (B, C, S)
        y = self.token_mlp_hidden(y)                 # (B, C, tokens_mlp_dim)
        y = self.activation(y)
        y = self.token_dropout(y, training=training)
        y = self.token_mlp_out(y)                    # (B, C, S)
        y = keras.ops.transpose(y, axes=(0, 2, 1))   # (B, S, C)
        x = inputs + y                               # residual

        # ---- Channel-mixing sub-block ----
        z = self.channel_norm(x)
        z = self.channel_mlp_hidden(z)               # (B, S, channels_mlp_dim)
        z = self.activation(z)
        z = self.channel_dropout(z, training=training)
        z = self.channel_mlp_out(z)                  # (B, S, C)
        output = x + z                               # residual

        return output

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """
        Compute the output shape, which is identical to the input shape.

        :param input_shape: Shape tuple of the input tensor.
        :type input_shape: Tuple[Optional[int], ...]
        :return: The unchanged input shape as a tuple.
        :rtype: Tuple[Optional[int], ...]
        """
        return tuple(input_shape)

    def get_config(self) -> Dict[str, Any]:
        """
        Get layer configuration for serialization.

        Returns ALL constructor parameters to ensure perfect reconstruction
        during model loading.

        :return: Dictionary containing the complete layer configuration.
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
