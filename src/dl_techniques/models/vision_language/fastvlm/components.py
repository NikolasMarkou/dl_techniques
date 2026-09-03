import keras
from keras import ops
from typing import Optional, Tuple, Dict, Any

from dl_techniques.layers.transformers import TransformerLayer
from dl_techniques.utils.keras_registration import register_dl_technique


@register_dl_technique("dl_techniques.models.fastvlm.components")
class AttentionBlockVLM(keras.layers.Layer):
    """
    Attention over a spatial feature map, via flatten-to-sequence and back.

    Wraps :class:`~dl_techniques.layers.transformers.TransformerLayer` for
    convolutional feature maps: it flattens ``[H, W, C]`` to a sequence,
    runs one transformer block, and reshapes the result back to ``[H, W,
    C]``. Layer scale is applied inside the transformer's own residual
    branches, not to the block's output as a whole.

    Architecture:

    .. code-block:: text

        input [B, H, W, C]
               │
               ▼
        flatten to [B, H*W, C]
               │
               ▼
        ┌─────────────────────┐
        │  TransformerLayer    │  attention + FFN, layer
        │                      │  scale inside residuals
        └──────────┬───────────┘
                    ▼
        reshape to [B, H, W, C]
               │
               ▼
        output [B, H, W, C]

    :param dim: Feature dimension. Must be positive and divisible by
        ``num_heads``.
    :type dim: int
    :param num_heads: Number of attention heads. Defaults to ``8``.
    :type num_heads: int
    :param mlp_ratio: Expansion ratio for the transformer's FFN. Defaults
        to ``4.0``.
    :type mlp_ratio: float
    :param attention_type: Attention variant forwarded to
        ``TransformerLayer`` (``'multi_head'``, ``'window'``,
        ``'group_query'``, see
        :class:`~dl_techniques.layers.attention.factory.AttentionType`).
        Defaults to ``'multi_head'``, which carries no positional
        information: with no positional embedding added before this
        block, a ``'multi_head'`` block is exactly permutation-equivariant
        over the spatial grid (measured: max\\|f(Px) - Pf(x)\\| = 5.36e-07).
        :class:`FastVLM` itself defaults to ``'group_query'`` instead; this
        constructor keeps ``'multi_head'`` as its own default only so
        existing direct callers keep their weight paths.
    :type attention_type: str
    :param normalization_position: ``'pre'`` or ``'post'``. Defaults to
        ``'pre'``.
    :type normalization_position: str
    :param dropout_rate: Elementwise dropout rate applied by the internal
        transformer (FFN output and attention probabilities), in
        ``[0, 1]``. Defaults to ``0.0``.
    :type dropout_rate: float
    :param use_stochastic_depth: Whether the internal transformer applies
        per-sample stochastic depth on its attention and FFN branches,
        independent of ``dropout_rate``. Defaults to ``False``.
    :type use_stochastic_depth: bool
    :param stochastic_depth_rate: Drop-path rate when
        ``use_stochastic_depth`` is ``True``. Must be below ``1.0``.
        Defaults to ``0.1``.
    :type stochastic_depth_rate: float
    :param max_seq_len: RoPE table length, consumed only when
        ``attention_type`` is ``'group_query'`` (other attention types do
        not declare this key). Must be at least the flattened grid size
        ``H * W``. Defaults to ``2048`` (a 45x45 grid, covering inputs up
        to roughly 720px at this block's stage-3 downsampling). Fixed
        rather than derived from an input shape, so a non-trainable
        weight's shape does not vary with resolution and weights transfer
        across resolutions.
    :type max_seq_len: int
    :param use_layer_scale: Whether to apply learnable layer scaling.
        Defaults to ``True``.
    :type use_layer_scale: bool
    :param layer_scale_init: Initial value for layer scale parameters.
        Must be positive. Defaults to ``1e-4``.
    :type layer_scale_init: float
    :param kwargs: Additional keyword arguments for the ``Layer`` base
        class.

    Input shape:
        4D tensor with shape: ``(batch_size, height, width, channels)``

    Output shape:
        4D tensor with same shape as input.

    :ivar transformer: The internal ``TransformerLayer`` doing attention
        and FFN, and owning layer scale when enabled.
    :ivar height: Height extracted from the input shape at build time.
    :ivar width: Width extracted from the input shape at build time.

    Example:
        .. code-block:: python

            attn = AttentionBlockVLM(dim=256, num_heads=8)
            inputs = keras.Input(shape=(14, 14, 256))
            outputs = attn(inputs)  # (None, 14, 14, 256)
    """

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            mlp_ratio: float = 4.0,
            attention_type: str = 'multi_head',
            normalization_position: str = 'pre',
            dropout_rate: float = 0.0,
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            use_layer_scale: bool = True,
            layer_scale_init: float = 1e-4,
            max_seq_len: int = 2048,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be between 0 and 1, got {dropout_rate}")
        if layer_scale_init <= 0:
            raise ValueError(f"layer_scale_init must be positive, got {layer_scale_init}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.normalization_position = normalization_position
        self.dropout_rate = dropout_rate
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init = layer_scale_init
        self.max_seq_len = max_seq_len

        # Will be set in build
        self.height = None
        self.width = None

        # Layer scale is delegated to TransformerLayer, applied inside each
        # residual branch rather than to this block's whole output.
        # DECISION plan-2026-08-18T140459-7991552f/D-043: max_seq_len is forwarded
        # only for 'group_query' -- the attention factory raises on an undeclared kwarg for other types. See decisions.md.
        attention_args = (
            {'max_seq_len': self.max_seq_len}
            if attention_type == 'group_query' else None
        )
        self.transformer = TransformerLayer(
            hidden_size=dim,
            num_heads=num_heads,
            intermediate_size=int(dim * mlp_ratio),
            attention_type=attention_type,
            attention_args=attention_args,
            normalization_position=normalization_position,
            dropout_rate=dropout_rate,
            attention_dropout_rate=dropout_rate,
            use_stochastic_depth=use_stochastic_depth,
            stochastic_depth_rate=stochastic_depth_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init,
            activation='gelu',
            name='vision_transformer'
        )

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the attention block and extract spatial dimensions."""
        if self.built:
            return
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input, got {len(input_shape)}D")

        batch_size, height, width, channels = input_shape

        if channels != self.dim:
            raise ValueError(f"Input channels ({channels}) must match dim ({self.dim})")

        self.height = height
        self.width = width

        if height is not None and width is not None:
            seq_length = height * width
        else:
            seq_length = None

        transformer_input_shape = (batch_size, seq_length, channels)
        self.transformer.build(transformer_input_shape)

        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Forward pass through attention block."""
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        height = input_shape[1] if self.height is None else self.height
        width = input_shape[2] if self.width is None else self.width

        # [B, H, W, C] -> [B, H*W, C]
        x = ops.reshape(inputs, (batch_size, height * width, self.dim))
        x = self.transformer(x, training=training)
        # [B, H*W, C] -> [B, H, W, C]. Layer scale already ran inside the transformer.
        x = ops.reshape(x, (batch_size, height, width, self.dim))

        return x

    def compute_output_shape(
            self,
            input_shape: Tuple[Optional[int], ...]
    ) -> Tuple[Optional[int], ...]:
        """Output shape is identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'num_heads': self.num_heads,
            'mlp_ratio': self.mlp_ratio,
            'attention_type': self.attention_type,
            'normalization_position': self.normalization_position,
            'dropout_rate': self.dropout_rate,
            'use_stochastic_depth': self.use_stochastic_depth,
            'stochastic_depth_rate': self.stochastic_depth_rate,
            'use_layer_scale': self.use_layer_scale,
            'layer_scale_init': self.layer_scale_init,
            'max_seq_len': self.max_seq_len,
        })
        return config

# ---------------------------------------------------------------------
