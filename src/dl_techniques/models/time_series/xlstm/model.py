"""xLSTM language model: stacked mLSTM and sLSTM residual blocks over token
embeddings, producing vocabulary logits. The continuous-input forecasting
sibling that reuses the same blocks is :class:`forecaster.xLSTMForecaster`.

A classic LSTM's sigmoid gates are bounded in ``(0, 1)``, so a stored value
can only decay, never be overwritten. xLSTM replaces them with exponential
gating, stabilized by a running log-domain maximum, so a strong new input
can dominate older memory. The layers chosen by ``mlstm_ratio`` also get a
matrix memory instead of a scalar one: capacity scales with
``key_dim * value_dim`` instead of width, and retrieval looks like
attention but keeps a fixed-size state instead of a growing KV cache.

Every path through the model is recurrent or causally convolved, so there
is no mask to omit and no way to leak future tokens. The mLSTM recurrence
is parallelizable in the paper; this implementation steps both cell types
sequentially through ``keras.layers.RNN``, so any throughput claim belongs
to the paper, not this code. ``from_variant(..., pretrained=True)`` raises
``NotImplementedError`` — no checkpoints ship with this package.

References:
    - Beck et al., 2024. xLSTM: Extended Long Short-Term Memory.
      (https://arxiv.org/abs/2405.04517)
    - Hochreiter and Schmidhuber, 1997. Long Short-Term Memory. Neural Computation
      9(8), 1735-1780.
    - Katharopoulos et al., 2020. Transformers are RNNs: Fast Autoregressive
      Transformers with Linear Attention. (https://arxiv.org/abs/2006.16236)
    - Shazeer, 2020. GLU Variants Improve Transformer.
      (https://arxiv.org/abs/2002.05202)
"""

import keras
from keras import layers, initializers
from typing import Optional, Union, Any, Dict, Literal

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.time_series.xlstm_blocks import mLSTMBlock, sLSTMBlock
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------

@register_dl_technique("dl_techniques.models.xlstm.model")
class xLSTM(keras.Model):
    """Stack of mLSTM and sLSTM residual blocks mapping tokens to vocabulary logits.

    Architecture:

    .. code-block:: text

        tokens [B, T]
           |
        Embedding                 -> [B, T, embed_dim]
           | (dropout, optional)
        mLSTM block  x n_mlstm     (lower layers)
           |
        sLSTM block  x n_slstm     (upper layers)
           |
        final normalization
           |
        Dense (output_head)       -> [B, T, vocab_size]

    The first ``int(num_layers * mlstm_ratio)`` blocks are mLSTM; the rest are sLSTM.

    :param vocab_size: Size of the vocabulary.
    :type vocab_size: int
    :param embed_dim: Dimensionality of token embeddings.
    :type embed_dim: int
    :param num_layers: Total number of xLSTM blocks.
    :type num_layers: int
    :param mlstm_ratio: Fraction of layers that are mLSTM, in [0, 1]. Defaults to 0.5.
    :type mlstm_ratio: float
    :param mlstm_num_heads: Number of heads for mLSTM blocks. Defaults to 4.
    :type mlstm_num_heads: int
    :param mlstm_expansion_factor: Expansion factor for mLSTM. Defaults to 2.
    :type mlstm_expansion_factor: int
    :param slstm_forget_gate: sLSTM forget-gate activation, ``'sigmoid'`` or ``'exp'``. Defaults to ``'sigmoid'``.
    :type slstm_forget_gate: str
    :param ffn_type: FFN type for sLSTM blocks. Defaults to ``'swiglu'``.
    :type ffn_type: str
    :param ffn_expansion_factor: FFN expansion factor for sLSTM. Defaults to 2.
    :type ffn_expansion_factor: int
    :param normalization_type: Normalization layer type. Defaults to ``'layer_norm'``.
    :type normalization_type: str
    :param normalization_kwargs: Extra keyword arguments for the normalization layer.
    :type normalization_kwargs: dict, optional
    :param dropout_rate: Dropout rate for the FFN in sLSTM blocks. Defaults to 0.0.
    :type dropout_rate: float
    :param embedding_dropout_rate: Dropout rate applied after the embedding. Defaults to 0.0.
    :type embedding_dropout_rate: float
    :param kernel_initializer: Initializer for kernel weights.
    :param recurrent_initializer: Initializer for recurrent weights.
    :param bias_initializer: Initializer for bias weights.
    :param kernel_regularizer: Optional regularizer for kernel weights.
    :param recurrent_regularizer: Optional regularizer for recurrent weights.
    :param bias_regularizer: Optional regularizer for bias weights.
    :param kwargs: Additional arguments for the Keras ``Model`` base class.

    Input shape:
        2D integer tensor with shape: `(batch_size, sequence_length)`.

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, vocab_size)`.

    Example:
        ```python
        model = xLSTM(
            vocab_size=50000,
            embed_dim=512,
            num_layers=12,
            mlstm_ratio=0.5,
            mlstm_num_heads=8,
            ffn_type='swiglu',
            normalization_type='rms_norm'
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        tokens = keras.random.randint(0, 50000, shape=(4, 128))
        logits = model(tokens)
        print(logits.shape)  # (4, 128, 50000)
        ```
    """

    DEFAULT_MLSTM_RATIO: float = 0.5
    DEFAULT_MLSTM_NUM_HEADS: int = 4
    DEFAULT_FFN_TYPE: str = 'swiglu'
    DEFAULT_NORMALIZATION_TYPE: str = 'layer_norm'

    # Size variants scale embed_dim, num_layers and heads for the language model.
    MODEL_VARIANTS = {
        "small": {
            "embed_dim": 256,
            "num_layers": 6,
            "mlstm_ratio": 0.5,
            "mlstm_num_heads": 4,
            "ffn_expansion_factor": 2,
        },
        "base": {
            "embed_dim": 512,
            "num_layers": 12,
            "mlstm_ratio": 0.5,
            "mlstm_num_heads": 8,
            "ffn_expansion_factor": 2,
        },
        "large": {
            "embed_dim": 1024,
            "num_layers": 24,
            "mlstm_ratio": 0.5,
            "mlstm_num_heads": 16,
            "ffn_expansion_factor": 4,
        },
    }

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        mlstm_ratio: float = 0.5,
        mlstm_num_heads: int = 4,
        mlstm_expansion_factor: int = 2,
        slstm_forget_gate: Literal['sigmoid', 'exp'] = 'sigmoid',
        ffn_type: str = 'swiglu',
        ffn_expansion_factor: int = 2,
        normalization_type: str = 'layer_norm',
        normalization_kwargs: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.0,
        embedding_dropout_rate: float = 0.0,
        kernel_initializer: Union[str, initializers.Initializer] = 'glorot_uniform',
        recurrent_initializer: Union[str, initializers.Initializer] = 'orthogonal',
        bias_initializer: Union[str, initializers.Initializer] = 'zeros',
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        recurrent_regularizer: Optional[keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0 <= mlstm_ratio <= 1:
            raise ValueError(f"mlstm_ratio must be in [0, 1], got {mlstm_ratio}")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlstm_ratio = mlstm_ratio
        self.mlstm_num_heads = mlstm_num_heads
        self.mlstm_expansion_factor = mlstm_expansion_factor
        self.slstm_forget_gate = slstm_forget_gate
        self.ffn_type = ffn_type
        self.ffn_expansion_factor = ffn_expansion_factor
        self.normalization_type = normalization_type
        # Keeps the None sentinel for lossless round-trip; `or {}` applies only
        # at the create_normalization_layer call site below.
        self.normalization_kwargs = normalization_kwargs
        self.dropout_rate = dropout_rate
        self.embedding_dropout_rate = embedding_dropout_rate
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.bias_regularizer = bias_regularizer

        self.embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim,
            name='embedding',
        )

        if embedding_dropout_rate > 0:
            self.embedding_dropout = layers.Dropout(
                rate=embedding_dropout_rate,
                name='embedding_dropout',
            )
        else:
            self.embedding_dropout = None

        self.blocks = []
        num_mlstm = int(num_layers * mlstm_ratio)

        for i in range(num_layers):
            if i < num_mlstm:
                block = mLSTMBlock(
                    units=embed_dim,
                    expansion_factor=mlstm_expansion_factor,
                    num_heads=mlstm_num_heads,
                    normalization_type=normalization_type,
                    normalization_kwargs=normalization_kwargs,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f'mlstm_block_{i}',
                )
            else:
                block = sLSTMBlock(
                    units=embed_dim,
                    ffn_type=ffn_type,
                    ffn_expansion_factor=ffn_expansion_factor,
                    normalization_type=normalization_type,
                    normalization_kwargs=normalization_kwargs,
                    forget_gate_activation=slstm_forget_gate,
                    dropout_rate=dropout_rate,
                    kernel_initializer=kernel_initializer,
                    recurrent_initializer=recurrent_initializer,
                    bias_initializer=bias_initializer,
                    kernel_regularizer=kernel_regularizer,
                    recurrent_regularizer=recurrent_regularizer,
                    bias_regularizer=bias_regularizer,
                    name=f'slstm_block_{i}',
                )

            self.blocks.append(block)

        self.final_norm = create_normalization_layer(
            normalization_type=normalization_type,
            name='final_norm',
            **(self.normalization_kwargs or {})
        )

        self.output_head = layers.Dense(
            vocab_size,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name='output_head',
        )

    def build(self, input_shape) -> None:
        """Build every sublayer explicitly, before ``super().build()``, so weights restore on `.keras` load.

        :param input_shape: Shape of the token input, `(batch_size, sequence_length)`.
        """
        # Token shape [B, T] maps to embedded shape [B, T, embed_dim].
        embedded_shape = tuple(input_shape) + (self.embed_dim,)

        self.embedding.build(input_shape)
        if self.embedding_dropout is not None:
            self.embedding_dropout.build(embedded_shape)
        for block in self.blocks:
            block.build(embedded_shape)
        self.final_norm.build(embedded_shape)
        self.output_head.build(embedded_shape)

        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        mask: Optional[keras.KerasTensor] = None,
    ) -> keras.KerasTensor:
        """Run the forward pass through the xLSTM model.

        :param inputs: Integer tensor of token ids, shape `(batch_size, seq_len)`.
        :param training: Whether the call runs in training mode.
        :type training: bool, optional
        :param mask: Optional mask tensor.
        :return: Logits tensor of shape `(batch_size, seq_len, vocab_size)`.
        :rtype: keras.KerasTensor
        """
        x = self.embedding(inputs, training=training)

        if self.embedding_dropout is not None:
            x = self.embedding_dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training, mask=mask)

        x = self.final_norm(x, training=training)
        logits = self.output_head(x, training=training)

        return logits

    @classmethod
    def from_variant(
        cls,
        variant: str,
        vocab_size: int,
        pretrained: bool = False,
        **overrides: Any
    ) -> 'xLSTM':
        """
        Create an :class:`xLSTM` language model from a predefined size variant.

        Args:
            variant: One of ``"small"``, ``"base"``, ``"large"``.
            vocab_size: Vocabulary size (required; not part of the variant dict).
            pretrained: Must be False; pretrained weights are not provided.
            **overrides: Override / supply constructor arguments. These take
                precedence over the variant defaults.

        Returns:
            An :class:`xLSTM` instance.

        Raises:
            ValueError: If ``variant`` is not recognized.
            NotImplementedError: If ``pretrained=True`` (no checkpoints shipped).

        Example:
            >>> model = xLSTM.from_variant("small", vocab_size=50000)
        """
        if pretrained:
            raise NotImplementedError(
                "Pretrained xLSTM weights are not provided. "
                "Use pretrained=False and train from scratch."
            )
        if variant not in cls.MODEL_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Available variants: "
                f"{list(cls.MODEL_VARIANTS.keys())}"
            )

        config = cls.MODEL_VARIANTS[variant].copy()
        config['vocab_size'] = vocab_size
        config.update(overrides)

        logger.info(f"Creating xLSTM-{variant.upper()} language model")

        return cls(**config)

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration of the model."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'num_layers': self.num_layers,
            'mlstm_ratio': self.mlstm_ratio,
            'mlstm_num_heads': self.mlstm_num_heads,
            'mlstm_expansion_factor': self.mlstm_expansion_factor,
            'slstm_forget_gate': self.slstm_forget_gate,
            'ffn_type': self.ffn_type,
            'ffn_expansion_factor': self.ffn_expansion_factor,
            'normalization_type': self.normalization_type,
            'normalization_kwargs': self.normalization_kwargs,
            'dropout_rate': self.dropout_rate,
            'embedding_dropout_rate': self.embedding_dropout_rate,
            'kernel_initializer': keras.initializers.serialize(
                initializers.get(self.kernel_initializer)
            ),
            'recurrent_initializer': keras.initializers.serialize(
                initializers.get(self.recurrent_initializer)
            ),
            'bias_initializer': keras.initializers.serialize(
                initializers.get(self.bias_initializer)
            ),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer': keras.regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'xLSTM':
        """Create a model from a configuration, deserializing initializers and regularizers first.

        :param config: Configuration dict as returned by :meth:`get_config`.
        :return: A reconstructed :class:`xLSTM` instance.
        """
        config = dict(config)
        for key in ("kernel_initializer", "recurrent_initializer",
                    "bias_initializer"):
            if config.get(key) is not None:
                config[key] = keras.initializers.deserialize(config[key])
        for key in ("kernel_regularizer", "recurrent_regularizer",
                    "bias_regularizer"):
            if config.get(key) is not None:
                config[key] = keras.regularizers.deserialize(config[key])
        return cls(**config)

# ---------------------------------------------------------------------


def create_xlstm(
    vocab_size: int,
    embed_dim: int,
    num_layers: int,
    mlstm_ratio: float = 0.5,
    mlstm_num_heads: int = 4,
    ffn_type: str = 'swiglu',
    normalization_type: str = 'layer_norm',
    **kwargs: Any
) -> xLSTM:
    """
    Factory for :class:`xLSTM` (language model).

    Thin config-driven constructor wrapper following the repo factory
    convention (mirrors ``create_xlstm_forecaster``). All additional
    constructor arguments are forwarded via ``**kwargs``.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Dimensionality of token embeddings.
        num_layers: Total number of xLSTM blocks.
        mlstm_ratio: Fraction of layers that are mLSTM. Defaults to 0.5.
        mlstm_num_heads: Number of mLSTM heads. Defaults to 4.
        ffn_type: FFN type for sLSTM blocks. Defaults to ``'swiglu'``.
        normalization_type: Normalization layer type. Defaults to
            ``'layer_norm'``.
        **kwargs: Forwarded to the :class:`xLSTM` constructor.

    Returns:
        A configured :class:`xLSTM` instance.
    """
    return xLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        mlstm_ratio=mlstm_ratio,
        mlstm_num_heads=mlstm_num_heads,
        ffn_type=ffn_type,
        normalization_type=normalization_type,
        **kwargs
    )

# ---------------------------------------------------------------------
