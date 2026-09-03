"""
Relational Graph Transformer, built by the `RELGT` class, with size presets in
`create_relgt_model`.

A relational database is a graph whose nodes come from different tables and whose
edges encode foreign keys and timestamps. RELGT keeps that structure instead of
flattening it into one node vector: each sampled node is split into five elements
(its features, table type, hop distance from the seed, time relative to the seed,
and structural position) embedded separately and summed, so attention can weigh
them as separable signals. Cost stays independent of database size because full
self-attention runs only inside a small sampled subgraph around each seed node;
each block also cross-attends the seed to a bank of learnable global centroids
that act as a database-wide codebook.

A block's local-attention output and its local/centroid summary are both carried
forward as a `(B, K, E)` token sequence, so stacking blocks adds real depth
instead of collapsing to a one-token sequence after the first block. The seed
node is assumed to be index 0 of `node_features`, a contract of the subgraph
sampler that the model itself does not check.

References:
    - Dwivedi et al., 2025. Relational Graph Transformer.
      (https://arxiv.org/abs/2505.10960)
    - Fey et al., 2023. Relational Deep Learning: Graph Representation Learning on
      Relational Databases. (https://arxiv.org/abs/2312.04615)
    - Dwivedi & Bresson, 2020. A Generalization of Transformer Networks to Graphs.
      (https://arxiv.org/abs/2012.09699)
    - Rampasek et al., 2022. Recipe for a General, Powerful, Scalable Graph
      Transformer. (https://arxiv.org/abs/2205.12454)
    - Jaegle et al., 2021. Perceiver: General Perception with Iterative Attention.
      (https://arxiv.org/abs/2103.03206)
"""

import keras
from keras import ops, layers
from typing import Optional, Dict, Any

# ---------------------------------------------------------------------
# local imports
# ---------------------------------------------------------------------

from dl_techniques.utils.logger import logger
from dl_techniques.layers.ffn import create_ffn_layer
from dl_techniques.layers.graphs.relational_graph_transformer_blocks import (
    RELGTTransformerBlock, RELGTTokenEncoder)
from dl_techniques.utils.model_build import materialize_sublayers
from dl_techniques.utils.keras_registration import register_dl_technique

# ---------------------------------------------------------------------


@register_dl_technique("dl_techniques.models.relgt.model")
class RELGT(keras.Model):
    """
    Multi-element tokenization plus hybrid local-global transformer blocks for
    predictive modeling on relational (multi-table) data.

    Architecture:

    .. code-block:: text

        features, types, hops, times, adjacency
                        │
                        ▼
        ┌───────────────────────────┐
        │  RELGTTokenEncoder         │  five embeddings, summed
        │  [B, K, E] local tokens    │
        └────────────┬───────────────┘
                      ▼
        ┌───────────────────────────┐
        │  TransformerBlock x N     │  local self-attn + centroid
        │  tokens in [B, K, E]      │  cross-attn, tokens chained
        │  out       (B, E) summary │
        └────────────┬───────────────┘
                      ▼
        ┌───────────────────────────┐
        │  Prediction head (FFN)     │
        └────────────┬───────────────┘
                      ▼
        output [B, output_dim]

    Variants (``MODEL_VARIANTS``, used by :func:`create_relgt_model`):

    .. code-block:: text

        name          embed_dim  heads  centroids  ffn_dim  blocks
        small                64      2         16      128       1
        base                512      4       4096     1024       1
        repo_medium         256      8         64      512       4

    :param output_dim: Dimension of the final output, e.g. number of classes.
        Must be positive.
    :type output_dim: int
    :param problem_type: ``'classification'`` or ``'regression'``; sets the
        final activation.
    :type problem_type: str
    :param embedding_dim: Main embedding dimension used throughout the model.
        Defaults to ``128``.
    :type embedding_dim: int
    :param num_node_types: Number of unique table/entity types. Defaults to ``10``.
    :type num_node_types: int
    :param max_hops: Maximum hop distance encoded. Defaults to ``2``.
    :type max_hops: int
    :param gnn_pe_dim: Dimension of the GNN positional encoding. Defaults to ``32``.
    :type gnn_pe_dim: int
    :param gnn_pe_layers: Number of GNN positional-encoding layers. Defaults to ``2``.
    :type gnn_pe_layers: int
    :param num_transformer_blocks: Number of stacked transformer blocks.
        Defaults to ``2``. Each block broadcast-adds its summary onto every
        token, so that token-invariant component grows relative to the
        token-varying signal with depth (measured untrained: ratio 0.23-0.49
        at block 0, past 1.0 by block 4 of an 8-block model). Only ``> 0`` is
        validated; re-measure the ratio before going past the 4-block depth
        this file's own ``repo_medium`` preset ships.
    :type num_transformer_blocks: int
    :param num_heads: Number of attention heads. Defaults to ``4``.
    :type num_heads: int
    :param num_global_centroids: Number of learnable global centroid tokens.
        Defaults to ``64``.
    :type num_global_centroids: int
    :param ffn_dim: Hidden dimension of the FFN blocks. Defaults to ``256``.
    :type ffn_dim: int
    :param dropout_rate: Dropout rate, in ``[0, 1)``. Defaults to ``0.1``.
    :type dropout_rate: float
    :param ffn_type: FFN variant passed to :func:`create_ffn_layer`. Defaults
        to ``'mlp'``.
    :type ffn_type: str
    :param normalization_type: Normalization variant used throughout. Defaults
        to ``'layer_norm'``.
    :type normalization_type: str
    :param kwargs: Forwarded to ``keras.Model``.
    :raises ValueError: If ``output_dim`` or ``embedding_dim`` is not positive,
        if ``problem_type`` is invalid, or if ``num_transformer_blocks`` is
        not positive.

    Input shape:
        A dict of tensors matching :class:`RELGTTokenEncoder`'s inputs.

    Output shape:
        ``(batch_size, output_dim)``.

    Example:
        .. code-block:: python

            model = RELGT(
                output_dim=10,
                problem_type='classification',
                embedding_dim=128,
                num_node_types=5,
                num_transformer_blocks=3,
                num_heads=8,
            )
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'],
            )
    """

    #: DECISION plan-2026-08-23T091307-9a110062/D-464: ``base`` mirrors
    #: upstream argparse defaults for channels/heads/centroids/layers only.
    #: Do not "correct" the unresolved gnn_pe_dim/pos_enc_dim mismatch. See decisions.md.
    MODEL_VARIANTS: Dict[str, Dict[str, Any]] = {
        # Repo-original. Smallest tier; used by the package's own smoke tests.
        "small": {
            "embedding_dim": 64,
            "num_heads": 2,
            "num_global_centroids": 16,
            "ffn_dim": 128,
            "num_transformer_blocks": 1,
        },
        # snap-stanford/relgt main_node_ddp.py argparse defaults.
        "base": {
            "embedding_dim": 512,
            "num_heads": 4,
            "num_global_centroids": 4096,
            "ffn_dim": 1024,
            "num_transformer_blocks": 1,
        },
        # Repo-original. Deeper (4 blocks) but narrower than `base`; the 4-block
        # depth is the ceiling documented at SUMMARY_BROADCAST_SCALE, do not raise
        # it without re-measuring.
        "repo_medium": {
            "embedding_dim": 256,
            "num_heads": 8,
            "num_global_centroids": 64,
            "ffn_dim": 512,
            "num_transformer_blocks": 4,
        },
    }

    def __init__(
            self,
            output_dim: int,
            problem_type: str = "classification",
            embedding_dim: int = 128,
            num_node_types: int = 10,
            max_hops: int = 2,
            gnn_pe_dim: int = 32,
            gnn_pe_layers: int = 2,
            num_transformer_blocks: int = 2,
            num_heads: int = 4,
            num_global_centroids: int = 64,
            ffn_dim: int = 256,
            dropout_rate: float = 0.1,
            ffn_type: str = 'mlp',
            normalization_type: str = 'layer_norm',
            **kwargs,
    ):
        super().__init__(**kwargs)

        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        if problem_type not in ["classification", "regression"]:
            raise ValueError(f"problem_type must be 'classification' or 'regression', got {problem_type}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_transformer_blocks <= 0:
            raise ValueError(f"num_transformer_blocks must be positive, got {num_transformer_blocks}")

        self.output_dim = output_dim
        self.problem_type = problem_type
        self.embedding_dim = embedding_dim
        self.num_node_types = num_node_types
        self.max_hops = max_hops
        self.gnn_pe_dim = gnn_pe_dim
        self.gnn_pe_layers = gnn_pe_layers
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.num_global_centroids = num_global_centroids
        self.ffn_dim = ffn_dim
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.normalization_type = normalization_type

        self.token_encoder = RELGTTokenEncoder(
            embedding_dim=embedding_dim,
            num_node_types=num_node_types,
            max_hops=max_hops,
            gnn_pe_dim=gnn_pe_dim,
            gnn_pe_layers=gnn_pe_layers,
            dropout_rate=dropout_rate,
            normalization_type=normalization_type,
            name="TokenEncoder",
        )

        self.transformer_blocks = []
        for i in range(num_transformer_blocks):
            block = RELGTTransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_global_centroids=num_global_centroids,
                ffn_dim=ffn_dim,
                dropout_rate=dropout_rate,
                ffn_type=ffn_type,
                normalization_type=normalization_type,
                return_tokens=True,
                name=f"TransformerBlock_{i}",
            )
            self.transformer_blocks.append(block)

        self.seed_encoder = layers.Dense(
            embedding_dim,
            name="SeedEncoder"
        )

        final_activation = "softmax" if problem_type == "classification" else None
        self.prediction_head = keras.Sequential([
            create_ffn_layer(
                ffn_type,
                hidden_dim=ffn_dim,
                output_dim=ffn_dim,
                dropout_rate=dropout_rate,
                name="PredictionFFN"
            ),
            layers.Dense(output_dim, activation=final_activation, name="FinalOutput")
        ], name="PredictionHead")

    def build(self, input_shape: Any) -> None:
        """Materialize every sub-layer from ``input_shape``.

        Without this method RELGT inherits ``Layer.build``, which marks the
        model built while every sub-layer is still unbuilt -- Keras warns about
        exactly that at ``layers/layer.py:393``. The shared helper traces
        ``call()`` on symbolic inputs, so what gets built cannot drift from what
        gets called.

        :param input_shape: Shape (or nest of shapes) of the input to ``call``.
        :type input_shape: Any
        """
        if self.built:
            return
        materialize_sublayers(self, input_shape)
        super().build(input_shape)

    def call(
            self,
            inputs: Dict[str, keras.KerasTensor],
            training: Optional[bool] = None
    ) -> keras.KerasTensor:
        """Run the token encoder and transformer blocks, then the prediction head.

        :param inputs: Dict of tensors matching :class:`RELGTTokenEncoder`'s inputs.
        :type inputs: Dict[str, keras.KerasTensor]
        :param training: Whether the model is in training mode.
        :type training: Optional[bool]
        :return: Predictions of shape ``(batch_size, output_dim)``.
        :rtype: keras.KerasTensor
        """
        # Seed node is index 0 of node_features, per the subgraph sampler's contract.
        seed_node_features = inputs["node_features"][:, 0:1, :]
        seed_node_embedding = self.seed_encoder(seed_node_features)

        local_tokens = self.token_encoder(inputs, training=training)

        # DECISION plan-2026-08-14T183218-f4c612aa/D-009: chain the (B, K, E) tokens,
        # not the (B, E) summary alone -- that collapses every later block to one token.
        # See decisions.md.
        current_representation = None
        current_tokens = local_tokens

        for block in self.transformer_blocks:
            current_representation, current_tokens = block(
                [current_tokens, seed_node_embedding],
                training=training
            )

        if current_representation is None:
            # No transformer blocks: fall back to mean-pooling the local tokens.
            current_representation = ops.mean(local_tokens, axis=1)

        predictions = self.prediction_head(current_representation, training=training)

        return predictions

    def get_config(self) -> Dict[str, Any]:
        """Return the config needed to reconstruct this model.

        :return: Constructor arguments.
        :rtype: Dict[str, Any]
        """
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "problem_type": self.problem_type,
            "embedding_dim": self.embedding_dim,
            "num_node_types": self.num_node_types,
            "max_hops": self.max_hops,
            "gnn_pe_dim": self.gnn_pe_dim,
            "gnn_pe_layers": self.gnn_pe_layers,
            "num_transformer_blocks": self.num_transformer_blocks,
            "num_heads": self.num_heads,
            "num_global_centroids": self.num_global_centroids,
            "ffn_dim": self.ffn_dim,
            "dropout_rate": self.dropout_rate,
            "ffn_type": self.ffn_type,
            "normalization_type": self.normalization_type,
        })
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'RELGT':
        """Build a model from a config dict produced by :meth:`get_config`.

        :param config: Configuration dictionary.
        :type config: Dict[str, Any]
        :return: New model instance.
        :rtype: RELGT
        """
        return cls(**config)

# ---------------------------------------------------------------------

def create_relgt_model(
        output_dim: int,
        problem_type: str = "classification",
        model_size: str = "base",
        **kwargs
) -> RELGT:
    """Build an RELGT model from one of the named size presets.

    :param output_dim: Dimension of the final output.
    :type output_dim: int
    :param problem_type: ``'classification'`` or ``'regression'``.
    :type problem_type: str
    :param model_size: ``'small'``, ``'base'``, or ``'repo_medium'``. Only
        ``'base'`` has an upstream counterpart; see :attr:`RELGT.MODEL_VARIANTS`.
    :type model_size: str
    :param kwargs: Overrides merged into the preset config.
    :return: A configured model.
    :rtype: RELGT
    :raises ValueError: If ``model_size`` is not a known preset.
    """
    size_configs = RELGT.MODEL_VARIANTS

    if model_size not in size_configs:
        raise ValueError(f"model_size must be one of {list(size_configs.keys())}, got {model_size}")

    config = size_configs[model_size].copy()
    config.update(kwargs)
    config.update({
        "output_dim": output_dim,
        "problem_type": problem_type,
    })

    logger.info(f"Creating RELGT model with size='{model_size}' and config: {config}")

    return RELGT(**config)

# ---------------------------------------------------------------------
