"""
An efficient multi-query attention mechanism for mobile devices.

This layer implements Mobile Multi-Query Attention (MobileMQA), a specialized
version of Grouped Query Attention (GQA) optimized for mobile and edge hardware.
It inherits the core projection and attention logic from GQA but introduces
vision-specific optimizations.

Key Features & Differences from Standard GQA:
1.  **Multi-Query Structure**: Forces `num_kv_heads=1`. All query heads share a
    single key/value head, minimizing memory bandwidth for K/V loading.
2.  **Spatial Downsampling**: Optional depthwise convolution on K/V feature maps
    *before* attention. This reduces the sequence length of the key/value pairs,
    lowering the attention complexity from O(N^2) to O(N*M).
3.  **Learnable Residual**: Uses a specialized residual connection
    `x + lambda * Attention(x)` with a learnable scalar `lambda` initialized to 1.
4.  **No RoPE**: MobileMQA typically relies on explicit positional embeddings or
    CNN-induced locality, so Rotary Position Embeddings are disabled by default.

Architecture:
    Input [B, H, W, C]
           ↓
    Projections (Q, K, V) -- Shared K/V head
           ↓
    (Optional) Downsample K, V (Depthwise Conv stride 2)
           ↓
    Flatten & Broadcast K, V to match Q heads
           ↓
    Attention(Q, K, V)
           ↓
    Output Projection
           ↓
    Residual: Input + lambda * Output

References:
    - Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need."
    - Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models."
"""

import keras
from keras import ops
from typing import Tuple, Optional, Any, Dict, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from .group_query_attention import GroupedQueryAttention

# ---------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MobileMQA(GroupedQueryAttention):
    """
    Mobile Multi-Query Attention (MobileMQA) block.

    A specialized subclass of GroupedQueryAttention that enforces Multi-Query
    Attention (1 KV head), supports optional spatial downsampling for Key/Value
    projections, and utilizes a learnable residual connection.

    Args:
        dim: Integer, input/output dimension. Must be positive and divisible by num_heads.
        num_heads: Integer, number of attention heads.
        use_downsampling: Boolean, whether to use spatial downsampling (stride 2
            DepthwiseConv2D) for keys and values. Defaults to False.
        kernel_initializer: Initializer for kernels. Defaults to 'he_normal'.
        kernel_regularizer: Optional regularizer.
        **kwargs: Additional arguments passed to GroupedQueryAttention.

    Input shape:
        4D tensor with shape: `(batch_size, height, width, dim)`

    Output shape:
        4D tensor with shape: `(batch_size, height, width, dim)`
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        use_downsampling: bool = False,
        kernel_initializer: Union[str, keras.initializers.Initializer] = "he_normal",
        kernel_regularizer: Optional[keras.regularizers.Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        # Enforce MQA configuration (num_kv_heads=1) and disable RoPE
        # Use bias=True to match standard Mobile/CNN conventions (vs GQA default False)
        kwargs['dim'] = dim
        kwargs['num_heads'] = num_heads
        kwargs['num_kv_heads'] = 1  # MQA definition
        kwargs['rope_percentage'] = 0.0  # Disable RoPE
        kwargs['use_bias'] = kwargs.get('use_bias', True)
        kwargs['kernel_initializer'] = kernel_initializer
        kwargs['kernel_regularizer'] = kernel_regularizer

        super().__init__(**kwargs)

        self.use_downsampling = use_downsampling

        # Config for downsampling layer if used
        if self.use_downsampling:
            self.downsample = keras.layers.DepthwiseConv2D(
                kernel_size=3,
                strides=2,
                padding="same",
                depthwise_initializer=self.kernel_initializer,
                depthwise_regularizer=self.kernel_regularizer,
                name="downsample"
            )
        else:
            self.downsample = None

        # Lambda parameter will be created in build()
        self.lambda_param = None

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """
        Build the layer. Calls super().build() for GQA weights, then adds
        MobileMQA-specific components (downsample, lambda).
        """
        # Build standard GQA weights (w_q, w_k, w_v, w_o)
        super().build(input_shape)

        # Create learnable residual scalar
        self.lambda_param = self.add_weight(
            name="lambda",
            shape=(),
            initializer="ones",
            trainable=True,
            dtype=self.compute_dtype
        )

        # Build downsample layer if active
        if self.downsample is not None:
            # Downsample operates on projected K/V
            # Shape: (Batch, H, W, head_dim) since num_kv_heads=1
            # We construct the expected shape based on input rank
            if len(input_shape) == 4:
                # Dense projection keeps spatial dims: (B, H, W, head_dim)
                kv_shape = list(input_shape)
                kv_shape[-1] = self.head_dim  # Output of w_k/w_v
                self.downsample.build(tuple(kv_shape))
            else:
                # Should typically be 4D for this layer, but handle gracefully
                pass

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
        attention_mask: Optional[keras.KerasTensor] = None,
        return_attention_weights: bool = False
    ) -> Union[keras.KerasTensor, Tuple[keras.KerasTensor, keras.KerasTensor]]:
        """
        Forward pass of MobileMQA.

        Overrides GQA.call to inject spatial downsampling logic and
        apply the specific lambda-residual connection.
        """
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        height, width = input_shape[1], input_shape[2]

        # 1. Project Q, K, V using inherited Dense layers
        # Shapes: (B, H, W, total_head_dim)
        q = self.w_q(inputs, training=training)
        k = self.w_k(inputs, training=training)
        v = self.w_v(inputs, training=training)

        # 2. Optional Spatial Downsampling for K and V
        if self.downsample is not None:
            k = self.downsample(k, training=training)
            v = self.downsample(v, training=training)

            # Update shapes for flattening
            kv_shape = ops.shape(k)
            kv_height, kv_width = kv_shape[1], kv_shape[2]
            kv_len = kv_height * kv_width
        else:
            kv_len = height * width

        # 3. Flatten Spatial Dimensions
        # Q: (B, H*W, num_heads * head_dim)
        q = ops.reshape(q, (batch_size, height * width, self.num_heads, self.head_dim))

        # K, V: (B, KV_Len, 1 * head_dim) -> MQA has 1 KV head
        k = ops.reshape(k, (batch_size, kv_len, 1, self.head_dim))
        v = ops.reshape(v, (batch_size, kv_len, 1, self.head_dim))

        # 4. Transpose to (B, Num_Heads, Seq_Len, Head_Dim)
        q = ops.transpose(q, (0, 2, 1, 3))  # (B, H, S_q, D)
        k = ops.transpose(k, (0, 2, 1, 3))  # (B, 1, S_kv, D)
        v = ops.transpose(v, (0, 2, 1, 3))  # (B, 1, S_kv, D)

        # 5. Broadcast K/V to match Q heads (Grouped Broadcast)
        # Since num_kv_heads=1, num_groups == num_heads
        k = ops.repeat(k, self.num_heads, axis=1)
        v = ops.repeat(v, self.num_heads, axis=1)

        # 6. Attention Mechanism
        # (B, H, S_q, D) @ (B, H, D, S_kv) -> (B, H, S_q, S_kv)
        scale = ops.cast(1.0 / ops.sqrt(ops.cast(self.head_dim, 'float32')), k.dtype)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale

        # Note: Masking is typically not used in standard MobileMQA vision contexts,
        # but if passed, would need careful handling due to downsampling.
        # We omit mask logic here to strictly match the vision use-case or GQA super if needed.

        attn_weights = ops.softmax(scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        # 7. Weighted Sum
        out = ops.matmul(attn_weights, v)  # (B, H, S_q, D)

        # 8. Reshape Output
        out = ops.transpose(out, (0, 2, 1, 3))  # (B, S_q, H, D)
        out = ops.reshape(out, (batch_size, height, width, self.dim))

        # 9. Output Projection & Lambda Residual
        attention_output = self.w_o(out, training=training)

        # Specific MobileMQA residual: inputs + lambda * output
        output = inputs + self.lambda_param * attention_output

        if return_attention_weights:
            return output, attn_weights
        return output

    def get_config(self) -> Dict[str, Any]:
        """Return config with MobileMQA specific parameters."""
        config = super().get_config()
        # Remove GQA-specific fields that we hardcoded/derived to avoid duplication in init
        # or keep them if we want full transparency.
        # Ideally, we only return what __init__ accepts.

        # Filter out parameters set by the subclass __init__
        params_to_remove = ['num_kv_heads', 'rope_percentage']
        for param in params_to_remove:
            config.pop(param, None)

        config.update({
            "use_downsampling": self.use_downsampling,
        })
        return config

# ---------------------------------------------------------------------
