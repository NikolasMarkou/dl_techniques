"""
Shared building blocks for the Qwen decoders: the causal mask constructor used by
both `qwen3.py` and `qwen3_next.py`, and the hybrid `Qwen3NextBlock`.

`build_causal_attention_mask` exists because nothing further down the stack
manufactures causality on its own. `TransformerLayer` defaults its `attention_mask`
to `None`, and `GroupQueryAttention` and `GatedAttention` mask only with what they
are handed, so a decoder that forwards just the caller's padding mask — which is
what both Qwen models did before this helper — lets every token attend to its own
future. The helper works entirely in *block* semantics (`True` = suppress),
OR-combining the lower-triangular causal mask with the padding mask derived from
`attention_mask == 0`, and inverts once at the very end to the *attend* semantics
(`True` = may attend) the attention layers expect. Doing the inversion once on the
combined mask rather than per component is what keeps the polarity tractable; the
GPT-2 path in `layers/transformers/text_decoder.py` follows the same discipline, and
its causality is pinned in both directions by a positive and a negative test.

`Qwen3NextBlock` is four residual updates, not one: three gated linear-attention
sublayers followed by one gated softmax-attention sublayer, each with its own
pre-normalization, its own optional mixture-of-experts FFN, and its own optional
stochastic-depth gate. The asymmetry is deliberate — the linear sublayers summarize
the past into a fixed-size recurrent state at `O(L)` cost, and the single attention
sublayer supplies the exact global lookup a bounded summary cannot. Only that
sublayer holds a KV cache, so the 3:1 ratio is what caps cache memory at roughly a
quarter of a uniformly attentive stack.

The mask reaches only the attention sublayer. The three linear-attention sublayers
are called without it, and that is not an omission to be repaired: a gated linear
scan is a strictly left-to-right recurrence over causal depthwise convolutions and
cannot read forward, so causality holds there structurally. What does *not* hold is
padding exclusion — padded positions still enter the recurrent state — so with
left-padded batches the summary a token sees is contaminated, while right-padding
leaves every valid prefix's state correct.

Normalization defaults to `zero_centered_rms_norm`. Plain RMS normalization divides
by the root-mean-square without removing a mean, so a persistent additive offset in
the residual stream survives every layer and consumes dynamic range the scale
weights are calibrated against; the zero-centered variant subtracts it.

References:
    - Qwen Team, 2025. Qwen3 Technical Report.
    - Yang et al., 2024. Gated Linear Attention Transformers with Hardware-Efficient
      Training. (https://arxiv.org/abs/2312.06635)
    - Katharopoulos et al., 2020. Transformers are RNNs: Fast Autoregressive
      Transformers with Linear Attention. (https://arxiv.org/abs/2006.16236)
    - Zhang and Sennrich, 2019. Root Mean Square Layer Normalization.
      (https://arxiv.org/abs/1910.07467)
"""

import keras
from typing import Optional, List, Any, Dict, Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.moe import MoEConfig
from dl_techniques.layers.moe import MixtureOfExperts
from dl_techniques.layers.transformers import GatedLinearAttentionBlock
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.attention.gated_attention import GatedAttention
from dl_techniques.utils.masking import create_mask, combine_masks, MaskConfig


# ---------------------------------------------------------------------


def build_causal_attention_mask(
        hidden_states: keras.KerasTensor,
        attention_mask: Optional[keras.KerasTensor] = None,
) -> keras.KerasTensor:
    """Build the causal (+ optional padding) attention mask for a Qwen stack.

    Qwen3 and Qwen3Next are decoder-only causal LMs, but neither built a causal
    mask: ``call`` forwarded only the *padding* mask (``None`` unless a caller
    supplied one), ``TransformerLayer`` defaults ``attention_mask=None``, and
    ``GroupQueryAttention``/``GatedAttention`` mask only when one is given. Every
    token therefore attended to every future token, so ``task_type="generation"``
    trained next-token prediction on a model that had already seen the answer.

    This mirrors the GPT-2 path in
    ``layers/transformers/text_decoder.py:505-538``, whose causality is pinned
    in both directions by ``tests/test_models/test_gpt2/test_gpt2.py:186``
    (future does not affect past) and ``:639`` (the negative control: without
    the mask it *does* leak).

    :param hidden_states: Embedded sequence, shape ``(batch, seq_len, dim)``.
    :type hidden_states: keras.KerasTensor
    :param attention_mask: Optional padding mask, shape ``(batch, seq_len)``,
        ``1`` for real tokens and ``0`` for padding.
    :type attention_mask: keras.KerasTensor or None
    :return: Boolean mask of shape ``(batch, seq_len, seq_len)`` in ATTEND
        semantics — ``True`` means "may attend" — which is the convention the
        attention layers expect.
    :rtype: keras.KerasTensor
    """
    batch_size = keras.ops.shape(hidden_states)[0]
    seq_len = keras.ops.shape(hidden_states)[1]

    # Block semantics throughout (True = block), inverted once at the end.
    causal_mask = create_mask('causal', seq_len=seq_len, dtype='bool')
    causal_mask = keras.ops.expand_dims(causal_mask, axis=0)
    causal_mask = keras.ops.broadcast_to(
        causal_mask, (batch_size, seq_len, seq_len))

    if attention_mask is not None:
        padding_mask_1d = keras.ops.equal(attention_mask, 0)
        padding_mask_3d = create_mask(config=MaskConfig(
            mask_type='padding',
            dtype='bool',
            extra_params={'padding_mask': padding_mask_1d},
        ))
        combined = combine_masks(causal_mask, padding_mask_3d, combination='or')
    else:
        combined = causal_mask

    return keras.ops.logical_not(combined)


# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class Qwen3NextBlock(keras.layers.Layer):
    """
    Qwen3 Next transformer block implementing the exact architectural pattern.

    This block implements a sophisticated transformer architecture consisting of
    3 cascaded Gated DeltaNet layers followed by 1 Gated Attention layer. Each
    layer is preceded by Zero-Centered RMSNorm and followed by optional MoE
    processing, with residual connections throughout.

    **Intent**: Implement the next-generation Qwen3 transformer block that combines
    DeltaNet's efficient sequence modeling with gated attention mechanisms for
    enhanced representational power while maintaining computational efficiency.

    **Architecture Flow**:
    ```
    Input(shape=[batch_size, seq_len, dim])
           ↓
    [Block 1] RMSNorm → Gated DeltaNet → MoE → Residual
           ↓
    [Block 2] RMSNorm → Gated DeltaNet → MoE → Residual
           ↓
    [Block 3] RMSNorm → Gated DeltaNet → MoE → Residual
           ↓
    [Block 4] RMSNorm → Gated Attention → MoE → Residual
           ↓
    Output(shape=[batch_size, seq_len, dim])
    ```

    **Component Details**:
    - **Gated DeltaNet**: Efficient sequence modeling with gating mechanisms
    - **Gated Attention**: Multi-head attention with gating for selective focus
    - **Zero-Centered RMSNorm**: Enhanced stability normalization for large models
    - **MoE**: Optional Mixture of Experts for conditional computation
    - **Stochastic Depth**: Optional training regularization via random layer dropping

    The 3+1 structure allows for hierarchical feature learning where the DeltaNet
    layers capture sequential patterns at different levels before the attention
    layer performs global contextualization.

    Args:
        dim: Integer, model dimension size. Must be positive and typically
            divisible by num_heads for efficient attention computation.
        num_heads: Integer, number of attention heads. Must be positive and
            should divide evenly into dim for optimal head dimension.
        head_dim: Optional integer, dimension per attention head. If None,
            defaults to dim // num_heads. Must be positive if specified.
        max_seq_len: Integer, maximum sequence length for RoPE embeddings
            in the attention layer. Defaults to 4096.
        moe_config: Optional MoEConfig instance for Mixture of Experts layers.
            If provided, each sub-layer will be followed by MoE processing.
            Can also be a dictionary that will be converted to MoEConfig.
            Defaults to None (no MoE).
        normalization_type: String, type of normalization layer to use.
            Supported types: 'zero_centered_rms_norm', 'layer_norm', 'rms_norm',
            'band_rms', etc. Defaults to 'zero_centered_rms_norm' for stability.
        norm_eps: Float, epsilon value for numerical stability in normalization.
            Should be small positive value. Defaults to 1e-6.
        dropout_rate: Float, dropout rate for regularization. Must be in [0, 1].
            Applied within the sub-layers. Defaults to 0.0.
        use_stochastic_depth: Boolean, whether to apply stochastic depth
            regularization. When True, randomly drops entire sub-layers during
            training. Defaults to False.
        stochastic_depth_rate: Float, probability of dropping layers when
            use_stochastic_depth=True. Must be in [0, 1]. Defaults to 0.1.
        **kwargs: Additional arguments for Layer base class (name, trainable, etc.).

    Input shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`

    Output shape:
        3D tensor with shape: `(batch_size, sequence_length, dim)`
        Shape is preserved through all processing stages.

    Attributes:
        delta_norms: List of normalization layers for DeltaNet blocks.
        delta_layers: List of GatedLinearAttentionBlock layers.
        delta_moe_layers: List of optional MoE layers for DeltaNet blocks.
        attention_norm: Normalization layer for attention block.
        attention_layer: GatedAttention layer.
        attention_moe: Optional MoE layer for attention block.
        stochastic_depth_layers: List of StochasticDepth layers if enabled.

    Example:
        ```python
        from dl_techniques.layers.moe import MoEConfig, ExpertConfig, GatingConfig

        # Basic usage without MoE
        block = Qwen3NextBlock(
            dim=768,
            num_heads=12,
            max_seq_len=2048,
            dropout_rate=0.1
        )

        # With MoE configuration
        moe_config = MoEConfig(
            num_experts=8,
            expert_config=ExpertConfig(
                ffn_config={'type': 'swiglu', 'output_dim': 768}
            ),
            gating_config=GatingConfig(top_k=2)
        )

        advanced_block = Qwen3NextBlock(
            dim=1024,
            num_heads=16,
            max_seq_len=8192,
            moe_config=moe_config,
            use_stochastic_depth=True,
            stochastic_depth_rate=0.1
        )

        # Process sequences
        inputs = keras.Input(shape=(512, 768))
        outputs = block(inputs)
        model = keras.Model(inputs, outputs)
        ```

    Note:
        This implementation follows the composite layer pattern with explicit
        sub-layer building for robust serialization. All sub-layers are built
        in the build() method to ensure proper weight initialization during
        model loading.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            max_seq_len: int = 4096,
            moe_config: Optional[Any] = None,  # MoEConfig or dict
            normalization_type: str = "zero_centered_rms_norm",
            norm_eps: float = 1e-6,
            dropout_rate: float = 0.0,
            use_stochastic_depth: bool = False,
            stochastic_depth_rate: float = 0.1,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        # Validate inputs
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if head_dim is not None and head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {head_dim}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        if not (0.0 <= dropout_rate <= 1.0):
            raise ValueError(f"dropout_rate must be in [0, 1], got {dropout_rate}")
        if not (0.0 <= stochastic_depth_rate <= 1.0):
            raise ValueError(f"stochastic_depth_rate must be in [0, 1], got {stochastic_depth_rate}")
        if norm_eps <= 0:
            raise ValueError(f"norm_eps must be positive, got {norm_eps}")

        # Store configuration
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else dim // num_heads
        self.max_seq_len = max_seq_len
        self.normalization_type = normalization_type
        self.norm_eps = norm_eps
        self.dropout_rate = dropout_rate
        self.use_stochastic_depth = use_stochastic_depth
        self.stochastic_depth_rate = stochastic_depth_rate

        # Handle MoE config - convert dict to MoEConfig if needed
        if moe_config is not None:
            if isinstance(moe_config, dict):
                self.moe_config = MoEConfig.from_dict(moe_config)
            else:
                self.moe_config = moe_config
        else:
            self.moe_config = None

        # CREATE all sub-layers in __init__ (they are unbuilt)

        # 3x Gated DeltaNet layers with their normalization and MoE
        self.delta_norms = []
        self.delta_layers = []
        self.delta_moe_layers = []

        for i in range(3):
            # Pre-layer normalization
            delta_norm = create_normalization_layer(
                self.normalization_type,
                epsilon=self.norm_eps,
                name=f"delta_norm_{i}"
            )
            self.delta_norms.append(delta_norm)

            delta_layer = GatedLinearAttentionBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                max_seq_len=max_seq_len,
                name=f"gated_linear_attention_{i}"
            )
            self.delta_layers.append(delta_layer)

            # MoE layer after DeltaNet
            if self.moe_config is not None:
                delta_moe = MixtureOfExperts(
                    self.moe_config,
                    name=f"delta_moe_{i}"
                )
                self.delta_moe_layers.append(delta_moe)
            else:
                self.delta_moe_layers.append(None)

        # 1x Gated Attention layer with its normalization and MoE
        self.attention_norm = create_normalization_layer(
            self.normalization_type,
            epsilon=self.norm_eps,
            name="attention_norm"
        )

        self.attention_layer = GatedAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            dropout_rate=self.dropout_rate,
            name="gated_attention"
        )

        # MoE layer after attention
        if self.moe_config is not None:
            self.attention_moe = MixtureOfExperts(
                self.moe_config,
                name="attention_moe"
            )
        else:
            self.attention_moe = None

        # Stochastic depth for regularization
        self.stochastic_depth_layers: List[Optional[Any]] = []
        if use_stochastic_depth and stochastic_depth_rate > 0.0:
            for i in range(4):  # 3 delta + 1 attention
                stoch_depth = StochasticDepth(
                    drop_path_rate=stochastic_depth_rate,
                    name=f"stochastic_depth_{i}"
                )
                self.stochastic_depth_layers.append(stoch_depth)
        else:
            # Fill with None for consistent indexing
            self.stochastic_depth_layers = [None] * 4

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build all sub-layers for robust serialization.

        CRITICAL: Explicitly build each sub-layer to ensure weight variables
        exist before weight restoration during model loading.
        """
        # Build delta layers and their components
        for i in range(3):
            self.delta_norms[i].build(input_shape)
            self.delta_layers[i].build(input_shape)

            if self.delta_moe_layers[i] is not None:
                self.delta_moe_layers[i].build(input_shape)

            if self.stochastic_depth_layers[i] is not None:
                self.stochastic_depth_layers[i].build(input_shape)

        # Build attention layer and its components
        self.attention_norm.build(input_shape)
        self.attention_layer.build(input_shape)

        if self.attention_moe is not None:
            self.attention_moe.build(input_shape)

        if self.stochastic_depth_layers[3] is not None:
            self.stochastic_depth_layers[3].build(input_shape)

        # Always call parent build at the end
        super().build(input_shape)

    def call(
            self,
            inputs: keras.KerasTensor,
            attention_mask: Optional[keras.KerasTensor] = None,
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """
        Forward pass through the Qwen3Next block.

        Args:
            inputs: Input tensor of shape (batch_size, seq_len, dim)
            attention_mask: Optional attention mask for the attention layer
            training: Training mode flag for dropout and normalization

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        x = inputs

        # Process through 3x Gated DeltaNet layers
        for i in range(3):
            # Pre-normalization
            x_norm = self.delta_norms[i](x, training=training)

            # Gated DeltaNet
            delta_out = self.delta_layers[i](x_norm, training=training)

            # MoE if configured
            if self.delta_moe_layers[i] is not None:
                delta_out = self.delta_moe_layers[i](delta_out, training=training)

            # Apply stochastic depth if configured
            if self.stochastic_depth_layers[i] is not None:
                delta_out = self.stochastic_depth_layers[i](
                    delta_out, training=training
                )

            # Residual connection
            x = x + delta_out

        # Process through 1x Gated Attention layer
        # Pre-normalization
        x_norm = self.attention_norm(x, training=training)

        # Gated Attention
        attention_out = self.attention_layer(
            x_norm,
            attention_mask=attention_mask,
            training=training
        )

        # MoE if configured
        if self.attention_moe is not None:
            attention_out = self.attention_moe(attention_out, training=training)

        # Apply stochastic depth if configured
        if self.stochastic_depth_layers[3] is not None:
            attention_out = self.stochastic_depth_layers[3](
                attention_out, training=training
            )

        # Residual connection
        x = x + attention_out

        return x

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute output shape - identical to input shape."""
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "max_seq_len": self.max_seq_len,
            "moe_config": self.moe_config.to_dict() if self.moe_config else None,
            "normalization_type": self.normalization_type,
            "norm_eps": self.norm_eps,
            "dropout_rate": self.dropout_rate,
            "use_stochastic_depth": self.use_stochastic_depth,
            "stochastic_depth_rate": self.stochastic_depth_rate,
        })
        return config

# ------------------------------------------------------------------------