"""
Qwen3 Next Model Implementation
============================================

A complete implementation of the Qwen3 Next architecture following the correct block structure:
- Each block contains 3x Gated DeltaNet layers + 1x Gated Attention layer
- Each layer has its own Zero-Centered RMSNorm and MoE
- Proper residual connections throughout

Based on the architectural diagram showing the precise layer arrangement and connections.
"""

import keras
from typing import Optional, List, Any, Dict, Tuple

# ---------------------------------------------------------------------
# Local Imports
# ---------------------------------------------------------------------

from dl_techniques.layers.moe import MoEConfig
from dl_techniques.layers.moe import MixtureOfExperts
from dl_techniques.layers.gated_delta_net import GatedDeltaNet
from dl_techniques.layers.stochastic_depth import StochasticDepth
from dl_techniques.layers.norms import create_normalization_layer
from dl_techniques.layers.attention.gated_attention import GatedAttention


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
        delta_layers: List of GatedDeltaNet layers.
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

            delta_layer = GatedDeltaNet(
                dim=self.dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dropout_rate=self.dropout_rate,
                max_seq_len=max_seq_len,
                name=f"gated_delta_net_{i}"
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

# ---------------------------------------------------------------------