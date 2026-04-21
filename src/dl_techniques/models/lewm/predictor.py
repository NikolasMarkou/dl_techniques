"""
ARPredictor — autoregressive next-state predictor for LeWM.

Wraps a stack of `AdaLNZeroConditionalBlock` layers with:
- a learned positional embedding of shape (1, num_frames, input_dim),
- an input projection (Dense) from input_dim -> hidden_dim (or Identity),
- a conditioning projection (Dense) from input_dim -> hidden_dim,
- a final LayerNorm (elementwise-affine default),
- an output projection (Dense) from hidden_dim -> output_dim.

Upstream reference: `/tmp/lewm_source/module.py:Transformer, ARPredictor`.
"""

import keras
from keras import ops
from typing import Any, Dict, Optional, Tuple

from dl_techniques.layers.adaln_zero import AdaLNZeroConditionalBlock


@keras.saving.register_keras_serializable()
class ARPredictor(keras.layers.Layer):
    """Autoregressive predictor with AdaLN-zero conditional Transformer stack.

    :param num_frames: maximum sequence length for the learned positional
        embedding.
    :param depth: number of AdaLN-zero Transformer blocks.
    :param num_heads: attention heads per block.
    :param dim_head: per-head dimension.
    :param mlp_dim: MLP hidden dim within each block.
    :param input_dim: dim of the input embeddings (= model D).
    :param hidden_dim: internal working dim of the stack.
    :param output_dim: output embedding dim (defaults to input_dim).
    :param dropout: dropout rate inside blocks.
    :param emb_dropout: dropout applied after pos-embedding addition.
    :param kwargs: passthrough to `keras.layers.Layer`.
    """

    def __init__(
        self,
        num_frames: int,
        depth: int,
        num_heads: int,
        dim_head: int,
        mlp_dim: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_frames <= 0 or depth <= 0:
            raise ValueError(
                f"num_frames and depth must be positive. "
                f"Got num_frames={num_frames}, depth={depth}."
            )
        self.num_frames = num_frames
        self.depth = depth
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.mlp_dim = mlp_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.dropout_rate = dropout
        self.emb_dropout_rate = emb_dropout

        # Projections — identity (via None) when dims match.
        self.input_proj = (
            keras.layers.Dense(hidden_dim, name="input_proj")
            if input_dim != hidden_dim else None
        )
        self.cond_proj = (
            keras.layers.Dense(hidden_dim, name="cond_proj")
            if input_dim != hidden_dim else None
        )
        self.output_proj = (
            keras.layers.Dense(self.output_dim, name="output_proj")
            if hidden_dim != self.output_dim else None
        )

        self.emb_dropout = keras.layers.Dropout(emb_dropout, name="emb_dropout")

        self.blocks = [
            AdaLNZeroConditionalBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                mlp_dim=mlp_dim,
                dropout=dropout,
                name=f"block_{i}",
            )
            for i in range(depth)
        ]

        self.final_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="final_norm"
        )

    def build(self, input_shape: Any) -> None:
        """Build sublayers. `input_shape` is `[x_shape, c_shape]`."""
        if not (isinstance(input_shape, (list, tuple)) and len(input_shape) == 2):
            raise ValueError(
                f"ARPredictor expects input_shape=[x_shape, c_shape]. "
                f"Got {input_shape}"
            )
        x_shape, c_shape = input_shape

        # Learned pos embedding: (1, num_frames, input_dim).
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_frames, self.input_dim),
            initializer=keras.initializers.RandomNormal(stddev=1.0),
            trainable=True,
        )

        self.emb_dropout.build(x_shape)

        # Input/cond projection shapes.
        if self.input_proj is not None:
            self.input_proj.build(x_shape)
            hidden_x_shape = tuple(list(x_shape[:-1]) + [self.hidden_dim])
        else:
            hidden_x_shape = tuple(x_shape)

        if self.cond_proj is not None:
            self.cond_proj.build(c_shape)
            hidden_c_shape = tuple(list(c_shape[:-1]) + [self.hidden_dim])
        else:
            hidden_c_shape = tuple(c_shape)

        for block in self.blocks:
            block.build([hidden_x_shape, hidden_c_shape])

        self.final_norm.build(hidden_x_shape)

        if self.output_proj is not None:
            self.output_proj.build(hidden_x_shape)

        super().build(input_shape)

    def call(self, inputs, training: Optional[bool] = None) -> keras.KerasTensor:
        if not (isinstance(inputs, (list, tuple)) and len(inputs) == 2):
            raise ValueError(
                f"ARPredictor expects inputs=[x, c]. Got type={type(inputs)}."
            )
        x, c = inputs

        # Add positional embedding (slice to current sequence length).
        T = ops.shape(x)[1]
        pos = self.pos_embedding[:, :T, :]
        x = x + pos
        x = self.emb_dropout(x, training=training)

        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.cond_proj is not None:
            c = self.cond_proj(c)

        for block in self.blocks:
            x = block([x, c], training=training)

        x = self.final_norm(x)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x

    def compute_output_shape(self, input_shape: Any) -> Tuple[Optional[int], ...]:
        x_shape, _ = input_shape
        return tuple(list(x_shape[:-1]) + [self.output_dim])

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "num_frames": self.num_frames,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "dim_head": self.dim_head,
            "mlp_dim": self.mlp_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "dropout": self.dropout_rate,
            "emb_dropout": self.emb_dropout_rate,
        })
        return config
