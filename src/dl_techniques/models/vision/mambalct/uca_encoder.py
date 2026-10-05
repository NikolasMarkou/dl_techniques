"""Unified context-appearance encoder (ucaEncoder) for MambaLCT.

A hierarchical vision transformer in the HiViT lineage (Zhang et al., 2023)
as used by MambaLCT (Li et al., 2024): a 4x4-stride-4 convolutional stem
followed by two 2x2-stride-2 merging stages (total stride 16), with
transformer blocks per stage and a size-agnostic depthwise-convolution
conditional positional encoding (no fixed-length learned table, so template
and search inputs of different resolutions share one weight set).

Every attention / FFN / normalization choice routes through the library
factories; :class:`TransformerLayer` is direct-imported (that package has
no ``create_*`` by design). Context tokens ``c_p`` are NOT handled here:
the caller concatenates them with the patch tokens before calling, matching
paper Eq. 9 (``Concat(i_c, i_s, i_t)`` into joint attention).

References:
    - Li et al., 2024. MambaLCT: Boosting Tracking via Long-term Context
      State Space Model. (https://arxiv.org/abs/2412.13615)
    - Zhang et al., 2023. HiViT: Hierarchical Vision Transformer Meets
      Masked Image Modeling. (https://arxiv.org/abs/2205.14949)
    - Dosovitskiy et al., 2020. An Image is Worth 16x16 Words.
"""

import keras
from keras import ops
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.layers.norms.factory import create_normalization_layer
from dl_techniques.layers.transformers.transformer import TransformerLayer
from dl_techniques.utils.keras_registration import register_dl_technique
from dl_techniques.utils.logger import logger

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.mambalct.uca_encoder")
class UcaEncoder(keras.layers.Layer):
    """Hierarchical ViT encoder with shared weights across resolutions.

    :param stage_dims: Channel width per stage. Defaults to ``[128, 256, 512]``.
    :type stage_dims: List[int]
    :param stage_depths: Transformer blocks per stage. Defaults to ``[2, 2, 6]``.
    :type stage_depths: List[int]
    :param num_heads: Attention heads per stage. Defaults to ``[4, 8, 8]``.
    :type num_heads: List[int]
    :param mlp_ratio: FFN expansion ratio. Defaults to 4.0.
    :type mlp_ratio: float
    :param attention_type: Attention factory key. Defaults to ``"multi_head"``.
    :type attention_type: str
    :param ffn_type: FFN factory key. Defaults to ``"mlp"``.
    :type ffn_type: str
    :param normalization_type: Norm factory key. Defaults to ``"layer_norm"``.
    :type normalization_type: str
    :param norm_epsilon: Epsilon for all normalization layers. Defaults to 1e-5.
    :type norm_epsilon: float
    :param dropout_rate: Dropout rate. Defaults to 0.0.
    :type dropout_rate: float
    :param use_bias: Whether projections use bias. Defaults to True.
    :type use_bias: bool

    Input shape:
        4D image tensor ``(batch, height, width, 3)``.

    Output shape:
        3D token tensor ``(batch, tokens, stage_dims[-1])``.
    """

    def __init__(
        self,
        stage_dims: Optional[List[int]] = None,
        stage_depths: Optional[List[int]] = None,
        num_heads: Optional[List[int]] = None,
        mlp_ratio: float = 4.0,
        attention_type: str = "multi_head",
        ffn_type: str = "mlp",
        normalization_type: str = "layer_norm",
        norm_epsilon: float = 1e-5,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        dims = [128, 256, 512] if stage_dims is None else list(stage_dims)
        depths = [2, 2, 6] if stage_depths is None else list(stage_depths)
        heads = [4, 8, 8] if num_heads is None else list(num_heads)
        if not (len(dims) == len(depths) == len(heads)):
            raise ValueError(
                f"stage_dims ({dims}), stage_depths ({depths}) and num_heads "
                f"({heads}) must have equal length"
            )
        if any(d <= 0 for d in dims):
            raise ValueError(f"stage_dims must be positive, got {dims}")
        if any(d <= 0 for d in depths):
            raise ValueError(f"stage_depths must be positive, got {depths}")
        if any(h <= 0 for h in heads):
            raise ValueError(f"num_heads must be positive, got {heads}")
        if any(d % h != 0 for d, h in zip(dims, heads)):
            raise ValueError(
                f"each stage_dim must be divisible by its num_heads, got "
                f"{dims} / {heads}"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        if norm_epsilon <= 0:
            raise ValueError(f"norm_epsilon must be positive, got {norm_epsilon}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )

        self.stage_dims = dims
        self.stage_depths = depths
        self.num_heads = heads
        self.mlp_ratio = mlp_ratio
        self.attention_type = attention_type
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type
        self.norm_epsilon = norm_epsilon
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.embed_dim = dims[-1]

        self.stem = keras.layers.Conv2D(
            filters=dims[0], kernel_size=4, strides=4, padding="valid",
            use_bias=use_bias, name="stem",
        )
        self.stem_norm = create_normalization_layer(
            normalization_type, epsilon=norm_epsilon, name="stem_norm"
        )

        self.merges: List[keras.layers.Layer] = []
        self.merge_norms: List[keras.layers.Layer] = []
        self.cpe_convs: List[keras.layers.Layer] = []
        # Flat block list (never a list of lists: Keras tracks one level of
        # Layer containers, so nesting silently drops weights from `.keras`
        # archives while forward still works). `stage_block_offsets` partitions
        # it back into stages.
        self.blocks: List[TransformerLayer] = []
        for s, (dim, depth, nhead) in enumerate(zip(dims, depths, heads)):
            if s > 0:
                self.merges.append(
                    keras.layers.Conv2D(
                        filters=dim, kernel_size=2, strides=2, padding="valid",
                        use_bias=use_bias, name=f"merge_{s}",
                    )
                )
                self.merge_norms.append(
                    create_normalization_layer(
                        normalization_type, epsilon=norm_epsilon,
                        name=f"merge_norm_{s}",
                    )
                )
            self.cpe_convs.append(
                keras.layers.DepthwiseConv2D(
                    kernel_size=3, strides=1, padding="same", use_bias=False,
                    name=f"cpe_{s}",
                )
            )
            for i in range(depth):
                self.blocks.append(
                    TransformerLayer(
                        hidden_size=dim,
                        num_heads=nhead,
                        intermediate_size=int(dim * mlp_ratio),
                        attention_type=attention_type,  # type: ignore[arg-type]
                        normalization_type=normalization_type,  # type: ignore[arg-type]
                        attention_norm_args={"epsilon": norm_epsilon},
                        ffn_norm_args={"epsilon": norm_epsilon},
                        ffn_type=ffn_type,  # type: ignore[arg-type]
                        dropout_rate=dropout_rate,
                        attention_dropout_rate=dropout_rate,
                        use_bias=use_bias,
                        name=f"stage_{s}_block_{i}",
                    )
                )

        self.final_norm = create_normalization_layer(
            normalization_type, epsilon=norm_epsilon, name="final_norm"
        )

        logger.info(
            f"Created UcaEncoder stages={len(dims)} dims={dims} "
            f"depths={depths} heads={heads}"
        )

    def _run_stage(
        self,
        x: keras.KerasTensor,
        stage: int,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Run CPE plus transformer blocks over a feature map.

        :param x: Feature map ``(B, H, W, C)``.
        :param stage: Stage index.
        :param training: Training mode flag.
        :return: Encoded feature map ``(B, H, W, C)``.
        """
        x = x + self.cpe_convs[stage](x, training=training)
        batch = ops.shape(x)[0]
        height = ops.shape(x)[1]
        width = ops.shape(x)[2]
        dim = self.stage_dims[stage]
        tokens = ops.reshape(x, (batch, height * width, dim))
        start = sum(self.stage_depths[:stage])
        for block in self.blocks[start:start + self.stage_depths[stage]]:
            tokens = block(tokens, training=training)
        return ops.reshape(tokens, (batch, height, width, dim))

    def build(self, input_shape: Tuple[Optional[int], ...]) -> None:
        """Build the stem, merges, CPE convs, blocks, and final norm.

        :param input_shape: Shape ``(batch, height, width, channels)``.
        """
        if self.built:
            return
        if len(input_shape) != 4:
            raise ValueError(f"Expected 4D input shape, got {input_shape}")
        shape: Tuple[Optional[int], ...] = input_shape
        self.stem.build(shape)
        shape = self.stem.compute_output_shape(shape)
        self.stem_norm.build(shape)
        for s in range(len(self.stage_dims)):
            if s > 0:
                self.merges[s - 1].build(shape)
                shape = self.merges[s - 1].compute_output_shape(shape)
                self.merge_norms[s - 1].build(shape)
            self.cpe_convs[s].build(shape)
            feat = (shape[0], None, self.stage_dims[s])
            start = sum(self.stage_depths[:s])
            for block in self.blocks[start:start + self.stage_depths[s]]:
                block.build(feat)
            if shape[1] is not None and shape[2] is not None:
                shape = (shape[0], shape[1], shape[2], self.stage_dims[s])
            else:
                shape = (shape[0], None, None, self.stage_dims[s])
        self.final_norm.build((shape[0], None, self.stage_dims[-1]))
        super().build(input_shape)

    def call(
        self,
        inputs: keras.KerasTensor,
        training: Optional[bool] = None,
    ) -> keras.KerasTensor:
        """Encode an image into patch tokens.

        :param inputs: Image ``(B, H, W, 3)``.
        :param training: Training mode flag.
        :return: Tokens ``(B, N, stage_dims[-1])``.
        """
        x = self.stem(inputs, training=training)
        x = self.stem_norm(x, training=training)
        for s in range(len(self.stage_dims)):
            if s > 0:
                x = self.merges[s - 1](x, training=training)
                x = self.merge_norms[s - 1](x, training=training)
            x = self._run_stage(x, s, training=training)
        batch = ops.shape(x)[0]
        tokens = ops.reshape(x, (batch, -1, self.stage_dims[-1]))
        return self.final_norm(tokens, training=training)

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        """Compute token output shape from stored config.

        :param input_shape: Input shape ``(B, H, W, C)``.
        :return: Output shape ``(B, None, stage_dims[-1])``.
        """
        return (input_shape[0], None, self.stage_dims[-1])

    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization.

        :return: Dictionary containing all constructor arguments.
        """
        config = super().get_config()
        config.update({
            "stage_dims": list(self.stage_dims),
            "stage_depths": list(self.stage_depths),
            "num_heads": list(self.num_heads),
            "mlp_ratio": self.mlp_ratio,
            "attention_type": self.attention_type,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
            "norm_epsilon": self.norm_epsilon,
            "dropout_rate": self.dropout_rate,
            "use_bias": self.use_bias,
        })
        return config
